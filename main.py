from env import IoTSystem
import numpy as np
from water_fill import waterfilling
from Alternating_Optimization import AlternatingOptimization


def main():
    system = IoTSystem()
    power_Noise = system.sigma2

    iot_positions = np.array([
        [391.03, 433.78, 0], [465.23, 535.78, 0], [263.85, 164.67, 0], 
        [352.51, 636.99, 0], [365.74, 971.82, 0], [320.80, 406.66, 0], 
        [170.55, 385.23, 0], [407.96, 280.95, 0], [440.52, 443.79, 0], 
        [267.70, 926.15, 0]
    ])
    uav_position = np.array([350, 350, 100])

    channel_gains, distances, path_losses = system.calculate_cluster_gains(iot_positions, uav_position)

    # 均匀功率分配
    powers = np.full(len(iot_positions), system.p_total / len(iot_positions))
    total_rate_bps, individual_rates = system.calculate_communication_rate(channel_gains, powers)

    print(f"before:{total_rate_bps}, time:{8e6/total_rate_bps}")
    
    powers,waterlevel = waterfilling(np.array(channel_gains), system.p_total, power_Noise)
    print(f"powers:{powers},waterlevel:{waterlevel}")
    a, individual_rates = system.calculate_communication_rate(channel_gains, powers)
    print(f"after:{a}, individual_rates:{individual_rates}, time:{8e6/a}")

    improvement = (a - total_rate_bps) / total_rate_bps * 100
    print(f"improvement:{improvement}%")

    # alternating_optimization = AlternatingOptimization(system)
    # print(alternating_optimization.solve(channel_gains, max_iter=100, tol=1e-8))


if __name__ == "__main__":
    main()