import numpy as np



class AlternatingOptimization:
    """
    修正的功率分配求解器
    """
    
    def __init__(self, system):
        self.sys = system
    
    def update_omega(self, powers, channel_gains, omega_prev, max_iter=50, tol=1e-10):
        """
        更新ω_j，严格按照公式41
        ω_j = 1 + Σ(l_ji*p_ji)/(σ₀² + K*l_ji*p_ji*ω_j^(-1))
        """
        omega = max(omega_prev, 1.0)  # 确保ω >= 1
        
        for iteration in range(max_iter):
            numerator = 0.0
            for l_ji, p_ji in zip(channel_gains, powers):
                if p_ji > 1e-15 and l_ji > 1e-20:  # 数值保护
                    denominator = self.sys.sigma2 + self.sys.K * l_ji * p_ji / omega
                    if denominator > 1e-20:
                        numerator += (l_ji * p_ji) / denominator
            
            omega_new = 1.0 + numerator
            omega_new = max(omega_new, 1.0)  # 确保 ω >= 1
            
            if abs(omega_new - omega) < tol:
                return omega_new
            
            omega = omega_new
        
        return omega
    
    def waterfilling_exact(self, channel_gains, omega, tol=1e-10):
        """
        精确的注水算法，严格满足 ∑p_ji* = p_j^total
        公式: p_ji* = (μ_j - σ₀²*ω_j/(K*l_ji))₀^(p_j^max)
        约束: ∑p_ji* = p_j^total (等式!)
        """
        I_j = len(channel_gains)
        
        # 计算每个用户的噪声地板
        noise_floors = np.zeros(I_j)
        for i in range(I_j):
            if channel_gains[i] > 1e-20:
                noise_floors[i] = (self.sys.sigma2 * omega) / (self.sys.K * channel_gains[i])
            else:
                noise_floors[i] = 1e10  # 信道很差，设置很高的阈值
        
        def calculate_total_power(mu):
            """给定水位μ，计算总功率"""
            total = 0.0
            for i in range(I_j):
                power = max(0, mu - noise_floors[i])
                power = min(power, self.sys.p_max)  # 单节点约束
                total += power
            return total
        
        def calculate_powers(mu):
            """给定水位μ，计算功率分配"""
            powers = np.zeros(I_j)
            for i in range(I_j):
                power = max(0, mu - noise_floors[i])
                powers[i] = min(power, self.sys.p_max)
            return powers
        
        # 二分搜索找到满足等式约束的μ
        mu_min = np.max(noise_floors)  # 最小水位
        mu_max = mu_min + self.sys.p_total  # 初始最大水位
        
        # 扩展搜索范围确保可行性
        while calculate_total_power(mu_max) < self.sys.p_total:
            mu_max *= 2
            if mu_max > 1e6:  # 防止无限循环
                break
        
        # 二分搜索
        for iteration in range(100):
            mu_mid = (mu_min + mu_max) / 2
            total_power = calculate_total_power(mu_mid)
            
            if abs(total_power - self.sys.p_total) < tol:
                return calculate_powers(mu_mid)
            elif total_power < self.sys.p_total:
                mu_min = mu_mid
            else:
                mu_max = mu_mid
        
        # 返回最接近的解
        return calculate_powers((mu_min + mu_max) / 2)
    
    def calculate_rate(self, powers, channel_gains, omega):
        """
        计算系统速率，严格按照README.md公式36
        r_j^ag = B^ag * [Σlog₂(1 + K*l_ji*p_ji*ω_j^(-1)/σ₀²) + 
                        K*log₂(ω_j^(-1)) - K*log₂(e)*(1-ω_j^(-1))]
        """
        if omega < 1.0:
            omega = 1.0  # 数值保护
        
        # 第一项：用户速率项
        user_rate_sum = 0.0
        for l_ji, p_ji in zip(channel_gains, powers):
            if p_ji > 1e-15 and l_ji > 1e-20:
                sinr = (self.sys.K * l_ji * p_ji) / (omega * self.sys.sigma2)
                if sinr > 1e-15:
                    user_rate_sum += np.log2(1 + sinr)
        
        # 第二项和第三项：大系统修正项
        if omega > 1.0:
            correction_term = (self.sys.K * np.log2(1/omega) - 
                             self.sys.K * np.log2(np.e) * (1 - 1/omega))
        else:
            correction_term = 0.0
        
        total_rate_per_hz = user_rate_sum + correction_term
        return self.sys.B * total_rate_per_hz
    
    def solve(self, channel_gains, max_iter=100, tol=1e-8):
        """
        主求解算法：交替优化
        """
        I_j = len(channel_gains)
        
        # 初始化
        powers = np.full(I_j, min(self.sys.p_total/I_j, self.sys.p_max))
        # 严格满足总功率约束
        powers = powers * (self.sys.p_total / np.sum(powers))
        omega = 1.0
        
        history = {'rates': [], 'powers': [], 'omegas': [], 'convergence': []}
        
        print(f"\n🔄 开始交替优化求解:")
        print(f"  节点数: {I_j}")
        print(f"  初始总功率: {np.sum(powers):.6f} W")
        
        for iteration in range(max_iter):
            powers_prev = powers.copy()
            omega_prev = omega
            
            # Step 1: 固定功率，更新ω
            omega = self.update_omega(powers, channel_gains, omega)
            
            # Step 2: 固定ω，更新功率分配（注水算法）
            powers = self.waterfilling_exact(channel_gains, omega)
            
            # 计算当前速率
            current_rate = self.calculate_rate(powers, channel_gains, omega)
            
            # 记录历史
            history['rates'].append(current_rate)
            history['powers'].append(powers.copy())
            history['omegas'].append(omega)
            
            # 收敛性检查
            power_change = np.linalg.norm(powers - powers_prev)
            omega_change = abs(omega - omega_prev)
            history['convergence'].append(power_change + omega_change)
            
            # 约束验证
            total_power = np.sum(powers)
            max_node_power = np.max(powers)
            
            if iteration % 5 == 0 or iteration < 3:
                print(f"  第{iteration+1:2d}次: 速率={current_rate/1e3:8.2f} kbps, "
                      f"ω={omega:8.4f}, 总功率={total_power:.6f}W, "
                      f"最大节点={max_node_power:.4f}W")
            
            # 收敛判断
            if power_change < tol and omega_change < tol:
                print(f"  ✅ 算法在第{iteration+1}次迭代收敛")
                break
        
        return {
            'powers': powers,
            'omega': omega,
            'rate': current_rate,
            'history': history,
            'iterations': iteration + 1
        }
