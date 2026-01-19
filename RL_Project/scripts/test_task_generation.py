"""
测试任务生成功能
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import EnvironmentConfig
from environments.uav_env_dude import UAVEnvDUDe


def test_task_generation():
    """测试任务生成功能"""
    print("="*80)
    print("任务生成功能测试")
    print("="*80)
    
    # 创建环境
    env = UAVEnvDUDe(
        EnvironmentConfig.UE_CLUSTER_1,
        EnvironmentConfig.UE_CLUSTER_2,
        num_uavs=EnvironmentConfig.NUM_UAVS
    )
    
    print(f"\n✓ 环境创建成功")
    print(f"  - UE数量: {env.ue_num}")
    print(f"  - UAV数量: {env.uav_num}")
    print(f"  - 任务类型数: {env.task_types}")
    
    # 测试任务参数范围
    print(f"\n【任务参数范围】")
    print(f"  输入数据大小 (D_in): {env.task_D_in_min/1e6:.2f} ~ {env.task_D_in_max/1e6:.2f} Mbits")
    print(f"  输出数据大小 (D_out): {env.task_D_out/1e6:.2f} Mbits (固定)")
    print(f"  CPU周期数 (omega): {env.task_omega_min:.2e} ~ {env.task_omega_max:.2e} cycles")
    print(f"  最大时延 (tau): {env.task_tau_min:.2f} ~ {env.task_tau_max:.2f} s")
    
    # 测试任务生成
    print(f"\n【任务生成测试】")
    print(f"生成所有UE的任务...")
    env.generate_all_tasks()
    
    # 显示前5个UE的任务信息
    print(f"\n前5个UE的任务信息:")
    print(f"{'UE':<5} {'D_in(Mbits)':<15} {'D_out(Mbits)':<15} {'omega(cycles)':<20} {'s':<5} {'tau(s)':<10}")
    print("-"*80)
    for i in range(min(5, env.ue_num)):
        task = env.tasks[i]
        if task:
            print(f"{i:<5} {task['D_in']/1e6:<15.4f} {task['D_out']/1e6:<15.4f} "
                  f"{task['omega']:<20.2e} {task['s']:<5} {task['tau']:<10.2f}")
    
    # 测试任务统计
    print(f"\n【任务统计】")
    all_tasks = env.get_all_tasks()
    d_in_list = [t['D_in'] for t in all_tasks if t]
    omega_list = [t['omega'] for t in all_tasks if t]
    tau_list = [t['tau'] for t in all_tasks if t]
    
    print(f"  输入数据大小统计:")
    print(f"    平均值: {np.mean(d_in_list)/1e6:.4f} Mbits")
    print(f"    最小值: {np.min(d_in_list)/1e6:.4f} Mbits")
    print(f"    最大值: {np.max(d_in_list)/1e6:.4f} Mbits")
    
    print(f"\n  CPU周期数统计:")
    print(f"    平均值: {np.mean(omega_list):.2e} cycles")
    print(f"    最小值: {np.min(omega_list):.2e} cycles")
    print(f"    最大值: {np.max(omega_list):.2e} cycles")
    
    print(f"\n  最大时延统计:")
    print(f"    平均值: {np.mean(tau_list):.2f} s")
    print(f"    最小值: {np.min(tau_list):.2f} s")
    print(f"    最大值: {np.max(tau_list):.2f} s")
    
    # 测试覆盖范围检查
    print(f"\n【覆盖范围检查测试】")
    print(f"UAV覆盖半径: {env.uav_coverage_radius} m")
    print(f"MBS覆盖半径: {env.mbs_coverage_radius} m")
    
    # 检查前5个UE的覆盖情况
    print(f"\n前5个UE的覆盖情况:")
    print(f"{'UE':<5} {'MBS':<8} {'UAV1':<8} {'UAV2':<8} {'覆盖基站':<20}")
    print("-"*60)
    for i in range(min(5, env.ue_num)):
        covered_bs = env.get_covered_bs_list(i)
        mbs_covered, _ = env.check_coverage(i, 0)
        uav1_covered, _ = env.check_coverage(i, 1)
        uav2_covered, _ = env.check_coverage(i, 2) if env.uav_num >= 2 else (False, float('inf'))
        
        bs_names = []
        if 0 in covered_bs:
            bs_names.append("MBS")
        if 1 in covered_bs:
            bs_names.append("UAV1")
        if 2 in covered_bs:
            bs_names.append("UAV2")
        
        print(f"{i:<5} {'✓' if mbs_covered else '✗':<8} "
              f"{'✓' if uav1_covered else '✗':<8} "
              f"{'✓' if uav2_covered else '✗':<8} "
              f"{', '.join(bs_names) if bs_names else 'None':<20}")
    
    # 测试step函数中的任务生成
    print(f"\n【Step函数中的任务生成测试】")
    print("执行一个step，检查任务是否更新...")
    
    # 保存旧任务
    old_tasks = [task.copy() if task else None for task in env.tasks]
    
    # 执行step（使用零动作）
    dummy_actions = [np.zeros(env.get_action_dim()) for _ in range(env.uav_num)]
    _, _, _, info = env.step(dummy_actions)
    
    # 检查任务是否更新
    new_tasks = info['tasks']
    tasks_changed = False
    for i in range(env.ue_num):
        if old_tasks[i] is None or new_tasks[i] is None:
            continue
        if old_tasks[i]['D_in'] != new_tasks[i]['D_in']:
            tasks_changed = True
            break
    
    if tasks_changed:
        print("✓ 任务在每个时间步成功更新")
    else:
        print("⚠ 警告：任务可能没有更新")
    
    print(f"\n【任务信息在info中】")
    print(f"  info['tasks'] 包含 {len(info['tasks'])} 个任务")
    print(f"  第一个任务: D_in={info['tasks'][0]['D_in']/1e6:.4f} Mbits")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == '__main__':
    test_task_generation()

