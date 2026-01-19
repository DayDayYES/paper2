"""
干扰测试脚本 - 测试不同用户关联情况下的干扰大小
无人机固定不动，可手动修改关联矩阵
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import EnvironmentConfig
from environments.uav_env_dude import UAVEnvDUDe


def create_manual_association_matrix(ue_num, bs_num, association_type='uniform'):
    """
    创建手动关联矩阵
    
    Args:
        ue_num: UE数量
        bs_num: 基站数量（包括MBS）
        association_type: 关联类型
            - 'uniform': 均匀分布
            - 'cluster1_to_uav1': 簇1关联到UAV1，簇2关联到UAV2
            - 'all_to_mbs': 全部关联到MBS
            - 'manual': 返回空矩阵，需要手动填写
    
    Returns:
        b_ul: 上行关联矩阵 [bs_num, ue_num]
        b_dl: 下行关联矩阵 [bs_num, ue_num]
    """
    b_ul = np.zeros((bs_num, ue_num))
    b_dl = np.zeros((bs_num, ue_num))
    
    if association_type == 'uniform':
        # 均匀分布：每个UE随机关联到一个基站
        for n in range(ue_num):
            bs_ul = np.random.randint(0, bs_num)
            bs_dl = np.random.randint(0, bs_num)
            b_ul[bs_ul, n] = 1
            b_dl[bs_dl, n] = 1
    
    elif association_type == 'cluster1_to_uav1':
        # 簇1关联到UAV1，簇2关联到UAV2
        cluster1_size = len(EnvironmentConfig.UE_CLUSTER_1)
        for n in range(ue_num):
            if n < cluster1_size:
                # 簇1：上行和下行都关联到UAV1（索引1）
                b_ul[1, n] = 1
                b_dl[1, n] = 1
            else:
                # 簇2：上行和下行都关联到UAV2（索引2）
                b_ul[2, n] = 1
                b_dl[2, n] = 1
    
    elif association_type == 'all_to_mbs':
        # 全部关联到MBS（索引0）
        for n in range(ue_num):
            b_ul[0, n] = 1
            b_dl[0, n] = 1
    
    elif association_type == 'manual':
        # 返回空矩阵，需要手动填写
        pass
    
    return b_ul, b_dl


def compute_detailed_interference(env, b_ul, b_dl, p_bs, PL_bs_ue, PL_bs_bs):
    """
    计算详细的干扰分量
    
    Returns:
        detailed_interference: 包含各种干扰分量的字典
    """
    ue_num = env.ue_num
    bs_num = env.total_bs_num
    
    # 初始化详细干扰记录
    detailed_interference = {
        'ue_index': [],
        'ul_bs': [],  # 上行关联基站
        'dl_bs': [],  # 下行关联基站
        'I_D_ul': [],  # 上行：来自其他基站下行的干扰
        'I_U_ul': [],  # 上行：来自其他UE上行的干扰
        'I_self_ul': [],  # 上行：自干扰
        'I_total_ul': [],  # 上行：总干扰
        'I_D_dl': [],  # 下行：来自其他基站下行的干扰
        'I_U_dl': [],  # 下行：来自其他UE上行的干扰
        'I_total_dl': [],  # 下行：总干扰
        'SINR_ul': [],  # 上行SINR
        'SINR_dl': [],  # 下行SINR
        'rate_ul': [],  # 上行速率
        'rate_dl': []   # 下行速率
    }
    
    # 计算上行干扰（对每个UE）
    for n in range(ue_num):
        u = np.argmax(b_ul[:, n])  # 上行关联基站
        
        # 1. 来自其他基站下行传输的干扰 I_D_u,u
        I_D_u_u = 0
        for v in range(bs_num):
            if v != u:
                for n_prime in range(ue_num):
                    I_D_u_u += b_dl[v, n_prime] * p_bs[v] / PL_bs_bs[v, u]
        
        # 2. 来自其他UE上行传输的干扰 I_U_u,u
        I_U_u_u = 0
        for n_prime in range(ue_num):
            if n_prime != n:
                I_U_u_u += b_ul[u, n_prime] * env.P_ue / PL_bs_ue[u, n_prime]
        
        # 3. 全双工自干扰 I_self_u
        I_self_u = p_bs[u] / (10 ** (env.xi / 10))
        
        # 总上行干扰
        I_total_ul = I_D_u_u + I_U_u_u + I_self_u
        
        # 上行SINR和速率
        SINR_ul = (env.P_ue / PL_bs_ue[u, n]) / (I_total_ul + env.sigma)
        rate_ul = b_ul[u, n] * env.B * np.log2(1 + SINR_ul)
        
        # 记录
        detailed_interference['ue_index'].append(n)
        detailed_interference['ul_bs'].append(u)
        detailed_interference['I_D_ul'].append(I_D_u_u)
        detailed_interference['I_U_ul'].append(I_U_u_u)
        detailed_interference['I_self_ul'].append(I_self_u)
        detailed_interference['I_total_ul'].append(I_total_ul)
        detailed_interference['SINR_ul'].append(SINR_ul)
        detailed_interference['rate_ul'].append(rate_ul)
    
    # 计算下行干扰（对每个UE）
    for n in range(ue_num):
        v = np.argmax(b_dl[:, n])  # 下行关联基站
        
        # 1. 来自其他基站下行传输的干扰 I_D_n,n
        I_D_n_n = 0
        for v_prime in range(bs_num):
            if v_prime != v:
                for n_prime in range(ue_num):
                    if n_prime == n:
                        I_D_n_n += b_dl[v_prime, n_prime] * p_bs[v_prime] / PL_bs_ue[v_prime, n]
        
        # 2. 来自UE上行传输的干扰 I_U_n,n
        I_U_n_n = 0
        for n_prime in range(ue_num):
            if n_prime != n:
                d_ue_ue = np.linalg.norm(np.array(env.uecord[n]) - np.array(env.uecord[n_prime]))
                PL_ue_ue = (4 * np.pi * env.f_c * d_ue_ue / env.c) ** 2 * (10 ** (env.beta_nlos / 10))
                I_U_n_n += env.P_ue / PL_ue_ue
        
        # 总下行干扰
        I_total_dl = I_D_n_n + I_U_n_n
        
        # 下行SINR和速率
        SINR_dl = (p_bs[v] / PL_bs_ue[v, n]) / (I_total_dl + env.sigma)
        rate_dl = b_dl[v, n] * env.B * np.log2(1 + SINR_dl)
        
        # 记录
        detailed_interference['dl_bs'].append(v)
        detailed_interference['I_D_dl'].append(I_D_n_n)
        detailed_interference['I_U_dl'].append(I_U_n_n)
        detailed_interference['I_total_dl'].append(I_total_dl)
        detailed_interference['SINR_dl'].append(SINR_dl)
        detailed_interference['rate_dl'].append(rate_dl)
    
    return detailed_interference


def print_interference_summary(detailed_interference, env):
    """打印干扰摘要"""
    print("\n" + "="*100)
    print("干扰分析摘要")
    print("="*100)
    
    # 统计信息
    print(f"\n【总体统计】")
    print(f"  总UE数量: {len(detailed_interference['ue_index'])}")
    print(f"  总基站数量: {env.total_bs_num} (MBS + {env.uav_num} UAVs)")
    
    # 上行干扰统计
    print(f"\n【上行干扰统计】")
    print(f"  平均来自基站下行干扰 (I_D_ul): {np.mean(detailed_interference['I_D_ul']):.2e} W")
    print(f"  平均来自UE上行干扰 (I_U_ul):   {np.mean(detailed_interference['I_U_ul']):.2e} W")
    print(f"  平均自干扰 (I_self_ul):        {np.mean(detailed_interference['I_self_ul']):.2e} W")
    print(f"  平均总上行干扰:                {np.mean(detailed_interference['I_total_ul']):.2e} W")
    print(f"  平均上行SINR:                  {np.mean(detailed_interference['SINR_ul']):.2f} dB")
    print(f"  平均上行速率:                  {np.mean(detailed_interference['rate_ul'])/1e6:.2f} Mbps")
    
    # 下行干扰统计
    print(f"\n【下行干扰统计】")
    print(f"  平均来自基站下行干扰 (I_D_dl): {np.mean(detailed_interference['I_D_dl']):.2e} W")
    print(f"  平均来自UE上行干扰 (I_U_dl):   {np.mean(detailed_interference['I_U_dl']):.2e} W")
    print(f"  平均总下行干扰:                {np.mean(detailed_interference['I_total_dl']):.2e} W")
    print(f"  平均下行SINR:                  {np.mean(detailed_interference['SINR_dl']):.2f} dB")
    print(f"  平均下行速率:                  {np.mean(detailed_interference['rate_dl'])/1e6:.2f} Mbps")
    
    # 按基站统计
    print(f"\n【按基站统计】")
    for bs_idx in range(env.total_bs_num):
        bs_name = "MBS" if bs_idx == 0 else f"UAV{bs_idx}"
        ul_ues = [i for i, bs in enumerate(detailed_interference['ul_bs']) if bs == bs_idx]
        dl_ues = [i for i, bs in enumerate(detailed_interference['dl_bs']) if bs == bs_idx]
        print(f"  {bs_name}:")
        print(f"    上行关联UE数: {len(ul_ues)}")
        print(f"    下行关联UE数: {len(dl_ues)}")
        if ul_ues:
            avg_ul_interference = np.mean([detailed_interference['I_total_ul'][i] for i in ul_ues])
            print(f"    平均上行干扰: {avg_ul_interference:.2e} W")
        if dl_ues:
            avg_dl_interference = np.mean([detailed_interference['I_total_dl'][i] for i in dl_ues])
            print(f"    平均下行干扰: {avg_dl_interference:.2e} W")


def print_detailed_table(detailed_interference):
    """打印详细表格"""
    print("\n" + "="*100)
    print("详细干扰分析表")
    print("="*100)
    
    df = pd.DataFrame(detailed_interference)
    
    # 格式化显示
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    # 旧版本pandas不支持None，使用大整数代替
    try:
        pd.set_option('display.max_colwidth', None)
    except (ValueError, TypeError):
        pd.set_option('display.max_colwidth', 100)  # 使用大整数

    
    print("\n前10个UE的详细信息:")
    print(df.head(10).to_string(index=False))
    
    print("\n所有UE的统计信息:")
    print(df.describe().to_string())


def plot_interference_analysis(detailed_interference, save_path=None):
    """绘制干扰分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 上行干扰分量对比
    ax = axes[0, 0]
    ue_indices = detailed_interference['ue_index']
    ax.bar(ue_indices, detailed_interference['I_D_ul'], label='I_D (基站下行干扰)', alpha=0.7)
    ax.bar(ue_indices, detailed_interference['I_U_ul'], bottom=detailed_interference['I_D_ul'],
           label='I_U (UE上行干扰)', alpha=0.7)
    ax.bar(ue_indices, detailed_interference['I_self_ul'],
           bottom=np.array(detailed_interference['I_D_ul']) + np.array(detailed_interference['I_U_ul']),
           label='I_self (自干扰)', alpha=0.7)
    ax.set_xlabel('UE Index')
    ax.set_ylabel('Interference (W)')
    ax.set_title('Uplink Interference Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 下行干扰分量对比
    ax = axes[0, 1]
    ax.bar(ue_indices, detailed_interference['I_D_dl'], label='I_D (基站下行干扰)', alpha=0.7, color='orange')
    ax.bar(ue_indices, detailed_interference['I_U_dl'], bottom=detailed_interference['I_D_dl'],
           label='I_U (UE上行干扰)', alpha=0.7, color='green')
    ax.set_xlabel('UE Index')
    ax.set_ylabel('Interference (W)')
    ax.set_title('Downlink Interference Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. SINR分布
    ax = axes[1, 0]
    ax.plot(ue_indices, detailed_interference['SINR_ul'], 'o-', label='Uplink SINR', linewidth=2)
    ax.plot(ue_indices, detailed_interference['SINR_dl'], 's-', label='Downlink SINR', linewidth=2)
    ax.set_xlabel('UE Index')
    ax.set_ylabel('SINR (linear)')
    ax.set_title('SINR Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 速率分布
    ax = axes[1, 1]
    ax.plot(ue_indices, np.array(detailed_interference['rate_ul'])/1e6, 'o-', 
           label='Uplink Rate', linewidth=2)
    ax.plot(ue_indices, np.array(detailed_interference['rate_dl'])/1e6, 's-', 
           label='Downlink Rate', linewidth=2)
    ax.set_xlabel('UE Index')
    ax.set_ylabel('Data Rate (Mbps)')
    ax.set_title('Data Rate Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 干扰分析图已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    print("="*100)
    print("干扰测试脚本 - 手动设置关联矩阵")
    print("="*100)
    
    # 创建环境
    env = UAVEnvDUDe(
        EnvironmentConfig.UE_CLUSTER_1,
        EnvironmentConfig.UE_CLUSTER_2,
        num_uavs=EnvironmentConfig.NUM_UAVS
    )
    
    # 固定UAV位置（不移动）
    print(f"\n✓ 环境创建成功")
    print(f"  - UAV数量: {env.uav_num}")
    print(f"  - UE数量: {env.ue_num}")
    print(f"  - 基站数量: {env.total_bs_num} (1 MBS + {env.uav_num} UAVs)")
    
    # 显示当前UAV位置
    print(f"\n【当前UAV位置】")
    for i, uav in enumerate(env.uavs):
        pos = uav.get_position()
        print(f"  UAV{i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
    mbs_pos = env.mbs.get_position()
    print(f"  MBS: [{mbs_pos[0]:.2f}, {mbs_pos[1]:.2f}, {mbs_pos[2]:.2f}]")
    
    # ==================== 手动设置关联矩阵 ====================
    # 您可以在这里修改关联矩阵
    # b_ul[i, j] = 1 表示基站i为UE j提供上行服务
    # b_dl[i, j] = 1 表示基站i为UE j提供下行服务
    
    print(f"\n【设置关联矩阵】")
    print("="*100)
    print("关联矩阵说明:")
    print("  - 行索引: 基站 (0=MBS, 1=UAV1, 2=UAV2, ...)")
    print("  - 列索引: UE (0, 1, 2, ..., 19)")
    print("  - 值: 1表示关联，0表示不关联")
    print("  - 每列只能有一个1（每个UE只能关联一个基站）")
    print("="*100)
    
    # ========== 在这里修改关联矩阵 ==========
    # 示例1: 全部关联到MBS
    # b_ul, b_dl = create_manual_association_matrix(env.ue_num, env.total_bs_num, 'all_to_mbs')
    
    # 示例2: 簇1关联UAV1，簇2关联UAV2
    # b_ul, b_dl = create_manual_association_matrix(env.ue_num, env.total_bs_num, 'cluster1_to_uav1')
    
    # 示例3: 手动设置（解耦关联）
    b_ul = np.zeros((env.total_bs_num, env.ue_num))
    b_dl = np.zeros((env.total_bs_num, env.ue_num))
    
    # 手动设置上行关联（示例：前10个UE关联到UAV1，后10个关联到UAV2）
    for n in range(env.ue_num):
        if n < 10:
            b_ul[1, n] = 1  # 前10个UE上行关联到UAV1
        else:
            b_ul[2, n] = 1  # 后10个UE上行关联到UAV2
    
    # 手动设置下行关联（示例：前10个UE关联到UAV1，后10个关联到MBS）
    for n in range(env.ue_num):
        if n < 10:
            b_dl[1, n] = 1  # 前10个UE下行关联到UAV1
        else:
            b_dl[0, n] = 1  # 后10个UE下行关联到MBS
    
    # ========== 关联矩阵设置结束 ==========
    
    # 验证关联矩阵
    print("\n【关联矩阵验证】")
    for n in range(env.ue_num):
        ul_bs = np.argmax(b_ul[:, n])
        dl_bs = np.argmax(b_dl[:, n])
        ul_name = "MBS" if ul_bs == 0 else f"UAV{ul_bs}"
        dl_name = "MBS" if dl_bs == 0 else f"UAV{dl_bs}"
        if n < 5:  # 只显示前5个
            print(f"  UE{n}: 上行→{ul_name}, 下行→{dl_name}")
    print(f"  ... (共{env.ue_num}个UE)")
    
    # 设置基站功率
    p_bs = np.zeros(env.total_bs_num)
    p_bs[0] = env.mbs.P_uav  # MBS功率
    p_bs[1] = env.uavs[0].P_uav  # UAV1功率（最大功率）
    p_bs[2] = env.uavs[1].P_uav  # UAV2功率（最大功率）
    
    print(f"\n【基站功率设置】")
    print(f"  MBS: {p_bs[0]:.2f} W")
    print(f"  UAV1: {p_bs[1]:.2f} W")
    print(f"  UAV2: {p_bs[2]:.2f} W")
    
    # 计算距离和路径损耗
    print(f"\n【计算路径损耗...】")
    d_bs_ue, d_bs_bs = env.calculate_all_distances()
    PL_bs_ue, PL_bs_bs = env.compute_all_path_loss(d_bs_ue, d_bs_bs)
    print("✓ 路径损耗计算完成")
    
    # 计算详细干扰
    print(f"\n【计算干扰...】")
    detailed_interference = compute_detailed_interference(env, b_ul, b_dl, p_bs, PL_bs_ue, PL_bs_bs)
    print("✓ 干扰计算完成")
    
    # 打印结果
    print_interference_summary(detailed_interference, env)
    print_detailed_table(detailed_interference)
    
    # 绘制图形
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'figures')
    save_path = os.path.join(results_dir, 'interference_analysis.png')
    plot_interference_analysis(detailed_interference, save_path)
    
    # 保存详细数据到CSV
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'logs', 'interference_test.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(detailed_interference)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 详细数据已保存到: {csv_path}")
    
    print("\n" + "="*100)
    print("测试完成！")
    print("="*100)


if __name__ == '__main__':
    main()