#include <vector>
#include <cmath>
#include <fstream>
#include "imm_filter.hpp"

using namespace aimer;

// 生成含噪声的观测数据
std::vector<math::YpdCoord> generate_measurements(double dt, double total_time) {
    std::vector<math::YpdCoord> measurements;
    const int steps = total_time / dt;

    Eigen::Vector3d pos(2.0, 1.5, 5.0); 
    Eigen::Vector3d vel(0.0, 0.0, 0.0);
    Eigen::Vector3d acc(0.0, 0.0, 0.0);

    const double phase_duration = 10.0; // 每段持续10秒
    const double total_cycle_time = phase_duration * 5; // 5段

    for (int i = 0; i < steps; ++i) {
        double t = i * dt;
        double cycle_t = fmod(t, total_cycle_time); // 当前周期内的时间

        // 状态切换逻辑
        if (cycle_t < 10.0) {
            // 静止
            acc.setZero();
            vel.setZero();
        } else if (cycle_t < 20.0) {
            // 匀加速
            acc = Eigen::Vector3d(50, -30, 20);
        } else if (cycle_t < 30.0) {
            // 匀速
            acc.setZero();
        } else if (cycle_t < 40.0) {
            // 匀减速
            acc = Eigen::Vector3d(-50, 30, -20);
        } else {
            // 静止
            acc.setZero();
            vel.setZero();
        }

        // 运动学更新
        pos += vel * dt + 0.5 * acc * dt * dt;
        vel += acc * dt;

        // 加噪声的观测数据
        math::YpdCoord true_ypd = math::xyz_to_ypd(pos);
        math::YpdCoord meas_ypd;
        meas_ypd.yaw = true_ypd.yaw + 0.01 * (rand() / (double)RAND_MAX - 0.5);
        meas_ypd.pitch = true_ypd.pitch + 0.01 * (rand() / (double)RAND_MAX - 0.5);
        meas_ypd.dis = true_ypd.dis + 0.01 * (rand() / (double)RAND_MAX - 0.5);

        measurements.push_back(meas_ypd);
    }

    return measurements;
}

int main() {
    // 参数设置
    const double dt = 0.1;
    const double total_time = 50.0;
    const int steps = total_time / dt;

    // 生成测试数据
    auto measurements = generate_measurements(dt, total_time);

    // 初始化IMM滤波器
    IMMFilter imm_filter;
    imm_filter.init(measurements[0], 0.0);

    // 打开数据记录文件
    std::ofstream data_file("imm_results.csv");
    data_file << "time,true_x,true_y,true_z,est_x,est_y,est_z,prob_static,prob_cv,prob_ca\n";
    
    // 主滤波循环
    for(int i = 0; i < steps; ++i) {
        double t = i * dt;
        
        // 滤波器更新
        imm_filter.update(measurements[i], t, {}, {});
        
        // 获取数据
        Eigen::Vector3d true_pos = math::ypd_to_xyz(measurements[i]);
        Eigen::Vector3d est_pos = imm_filter.predict_pos(t);
        Eigen::Vector3d probs = imm_filter.get_model_probabilities();
        
        // 写入CSV
        data_file << std::fixed << std::setprecision(6)
                 << t << ","
                 << true_pos.x() << "," << true_pos.y() << "," << true_pos.z() << ","
                 << est_pos.x() << "," << est_pos.y() << "," << est_pos.z() << ","
                 << probs[0] << "," << probs[1] << "," << probs[2] << "\n";
    }
    data_file.close();
    return 0;
}