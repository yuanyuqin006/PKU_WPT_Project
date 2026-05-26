// solver.cpp —— 基于椭圆积分的偏心圆环互感
//
// 方法（直接用椭圆积分公式）：
//   半径 a 的圆形电流环置于 z=0 平面、圆心在原点，单位电流产生的方位向矢势：
//       A_phi(rho, z) = (mu0 / (pi * k)) * sqrt(a/rho) * [(1 - k^2/2)*K(k) - E(k)]
//       k^2 = 4 a rho / [(a + rho)^2 + z^2]
//   其中 K, E 分别为第一、二类完全椭圆积分（C++17：std::comp_ellint_1 / _2）。
//
//   半径 b 的接收环放在 z=h 平面、圆心横向偏移 d。沿接收环参数化 phi：
//       rho(phi) = sqrt(b^2 + d^2 + 2 b d cos(phi))
//   通过 Stokes 把通量化为环路线积分 Phi = ∮ A · dl，得到：
//       M_ring(a, b, h, d) = ∫_0^{2*pi} A_phi(rho(phi), h) * b * (b + d cos(phi)) / rho(phi) d phi
//
//   d = 0 时退化为经典 Maxwell 公式 M = mu0 * sqrt(a*b) * [(2/k - k)*K(k) - (2/k)*E(k)]。
//
// calc_M_total：把有厚度/多匝的线圈用径向离散化，双重循环遍历 (a_i, b_j) 并累加 M_ring。

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <limits>

// ---- 兜底：libc++ (Apple Clang) 未实现 C++17 特殊数学函数 ----
// 检测到标准库没有提供时，用 AGM 算法补一份 std::comp_ellint_1 / _2，
// 这样下面物理公式里写的 std::comp_ellint_* 在两种工具链上都能编译。
#ifndef __cpp_lib_math_special_functions
namespace std {
    inline double comp_ellint_1(double k) {
        constexpr double pi_v = 3.141592653589793238462643383279502884;
        const double ak = (k < 0 ? -k : k);
        if (ak >= 1.0) return std::numeric_limits<double>::infinity();
        if (ak == 0.0) return pi_v * 0.5;
        double a = 1.0;
        double b = sqrt(1.0 - ak * ak);
        for (int i = 0; i < 100; ++i) {
            const double an = 0.5 * (a + b);
            const double bn = sqrt(a * b);
            if ((a > b ? a - b : b - a) <= 1e-16 * an) { a = an; break; }
            a = an; b = bn;
        }
        return pi_v / (2.0 * a);
    }
    inline double comp_ellint_2(double k) {
        constexpr double pi_v = 3.141592653589793238462643383279502884;
        const double ak = (k < 0 ? -k : k);
        if (ak == 0.0) return pi_v * 0.5;
        if (ak >= 1.0) return 1.0;
        double a = 1.0;
        double b = sqrt(1.0 - ak * ak);
        double c = ak;
        double pow2 = 1.0;
        double s = c * c;            // 2^0 * c_0^2
        for (int i = 0; i < 100; ++i) {
            const double an = 0.5 * (a + b);
            const double bn = sqrt(a * b);
            c    = 0.5 * (a - b);
            pow2 *= 2.0;
            s    += pow2 * c * c;
            if ((a > b ? a - b : b - a) <= 1e-16 * an) { a = an; break; }
            a = an; b = bn;
        }
        const double K = pi_v / (2.0 * a);
        return K * (1.0 - 0.5 * s);
    }
}
#endif

constexpr double PI  = 3.14159265358979323846;
constexpr double MU0 = 4.0 * PI * 1e-7;            // H/m

// ---------------- 可调参数 ----------------
constexpr double A_INNER = 0.045;   // TX 内半径 (m)
constexpr double A_OUTER = 0.055;   // TX 外半径 (m)
constexpr int    N_A     = 5;       // TX 径向离散层数（视作匝数）

constexpr double B_INNER = 0.045;   // RX 内半径 (m)
constexpr double B_OUTER = 0.055;   // RX 外半径 (m)
constexpr int    N_B     = 5;       // RX 径向离散层数

constexpr int    N_PHI   = 360;     // 角度积分点数

// ---- 三线圈耦合 (TX—金属板—RX) 参数 ----
constexpr double F_OP    = 200e3;            // Qi WPT 典型工作频率 (Hz)
constexpr double OMEGA   = 2.0 * PI * F_OP;
constexpr double R_PLATE = 0.045;            // 金属板等效环路半径 (m)
constexpr double T_PLATE = 0.5e-3;           // 金属板厚度 (m)
constexpr double X0_FIXED_METAL = 0.0;       // 金属仿真中固定的偏心 (m, 共轴)
// ------------------------------------------

// 两个细丝圆环之间的互感 (单位 H)：
//   a, b 为发射 / 接收圆环半径 (m)
//   h    为轴向距离 (m)
//   d    为圆心横向偏心距 (m)
double calc_M_ring(double a, double b, double h, double d)
{
    const double dphi = 2.0 * PI / static_cast<double>(N_PHI);
    double sum = 0.0;

    for (int i = 0; i < N_PHI; ++i) {
        const double phi  = (i + 0.5) * dphi;      // 中点法则
        const double cphi = std::cos(phi);

        // 接收环上当前点到 TX 轴 (z 轴) 的径向距离
        const double rho  = std::sqrt(b * b + d * d + 2.0 * b * d * cphi);

        // 椭圆积分模数
        const double denom = (a + rho) * (a + rho) + h * h;
        const double k2    = 4.0 * a * rho / denom;
        const double k     = std::sqrt(k2);

        // C++17 标准库特殊函数：第一/第二类完全椭圆积分
        const double K = std::comp_ellint_1(k);
        const double E = std::comp_ellint_2(k);

        // A_phi(rho, h) / I  ——  矢势的方位向分量除以电流
        const double bracket    = (1.0 - 0.5 * k2) * K - E;
        const double Aphi_per_I = (MU0 / (PI * k)) * std::sqrt(a / rho) * bracket;

        // 通量积分被积函数： A_phi · (b̂_RX · φ̂_TX) * b dphi
        //                = A_phi(rho, h) * b * (b + d cos phi) / rho
        sum += Aphi_per_I * b * (b + d * cphi) / rho;
    }
    return sum * dphi;
}

// 有限截面/多匝线圈互感：把 a, b 各离散成 Na, Nb 个薄环并累加
double calc_M_total(double a_in, double a_out, int Na,
                    double b_in, double b_out, int Nb,
                    double h,    double d)
{
    double M = 0.0;
    for (int i = 0; i < Na; ++i) {
        const double ai = a_in + (a_out - a_in) * (i + 0.5) / static_cast<double>(Na);
        for (int j = 0; j < Nb; ++j) {
            const double bj = b_in + (b_out - b_in) * (j + 0.5) / static_cast<double>(Nb);
            M += calc_M_ring(ai, bj, h, d);
        }
    }
    return M;
}

// ---------------- 三线圈耦合修正 ----------------
// 趋肤深度 δ = sqrt(2 / (ω·μ·σ))
double skin_depth(double sigma)
{
    return std::sqrt(2.0 / (OMEGA * MU0 * sigma));
}

// 把 TX—中间金属薄板—RX 视为三个耦合环路：
//   V_m = jω·M_1m·I_TX   (金属板感生电压)
//   I_m = V_m / (R_m + jω·L_m)
//   V_R = jω·(M_12·I_TX + M_2m·I_m)
// 解出有效互感：
//   M_eff = M_12 - jω·M_1m·M_2m / (R_m + jω·L_m)
// 返回 |M_eff| 用于功率耦合可视化。
double calc_M_three_coil(double a_in, double a_out, int Na,
                         double b_in, double b_out, int Nb,
                         double h_axial, double x0,
                         double sigma, double z_plate)
{
    // 1) 直接互感 (TX → RX)，已有椭圆积分公式
    const double M12 = calc_M_total(a_in, a_out, Na, b_in, b_out, Nb, h_axial, x0);
    // 2) TX → 金属环 (假设金属板与 TX 共轴)
    const double M1m = calc_M_total(a_in, a_out, Na,
                                    R_PLATE, R_PLATE, 1,
                                    z_plate, 0.0);
    // 3) 金属环 → RX (轴向距 h-z_plate, 横向偏 x0)
    const double M2m = calc_M_total(R_PLATE, R_PLATE, 1,
                                    b_in, b_out, Nb,
                                    h_axial - z_plate, x0);

    // 4) 金属环自感：圆环近似 L = μ₀·R·(ln(8R/r_w) - 2)，r_w 取等效"导线半径"
    const double r_wire = 1.0e-3;
    const double Lm     = MU0 * R_PLATE * (std::log(8.0 * R_PLATE / r_wire) - 2.0);
    // 5) 金属环等效电阻：薄板片电阻 ρ/t 乘以环周长/宽度比 ≈ 2π
    const double Rm     = 2.0 * PI / (sigma * T_PLATE);

    const double wL = OMEGA * Lm;
    const double D  = Rm * Rm + wL * wL;
    // M_eff = M12 - jω·M_1m·M_2m / (R_m + jωL_m)
    const double Re_Meff = M12 - OMEGA * OMEGA * M1m * M2m * Lm / D;
    const double Im_Meff = -OMEGA * M1m * M2m * Rm / D;
    return std::sqrt(Re_Meff * Re_Meff + Im_Meff * Im_Meff);
}

int main()
{
    // 扫描：x0 = 横向偏心 (mm)，d = 轴向距离 (mm)
    std::vector<int> x0_mm, d_mm;
    for (int v = 0; v <= 40; ++v) x0_mm.push_back(v);
    for (int v = 1; v <= 30; ++v) d_mm.push_back(v);

    const std::size_t Nx = x0_mm.size();
    const std::size_t Nd = d_mm.size();

    std::vector<std::vector<double>> M(Nx, std::vector<double>(Nd, 0.0));

    std::cout << "Computing mutual inductance via elliptic integrals over "
              << Nx << " x " << Nd << " grid..." << std::endl;

    for (std::size_t i = 0; i < Nx; ++i) {
        for (std::size_t j = 0; j < Nd; ++j) {
            const double d_offset = x0_mm[i] * 1.0e-3;   // 横向 (m)
            const double h_axial  = d_mm[j]  * 1.0e-3;   // 轴向 (m)
            M[i][j] = calc_M_total(A_INNER, A_OUTER, N_A,
                                   B_INNER, B_OUTER, N_B,
                                   h_axial, d_offset);
        }
        std::cout << "  x0 = " << x0_mm[i] << " mm  done" << std::endl;
    }

    // 写出 JSON
    std::ofstream f("simulation_data.json");
    if (!f) {
        std::cerr << "Failed to open simulation_data.json." << std::endl;
        return 1;
    }

    f << "{\n";
    f << "  \"parameters\": {\n";
    f << "    \"method\": \"elliptic-integral (std::comp_ellint_1 / _2)\",\n";
    f << "    \"a_inner_m\": " << A_INNER << ",\n";
    f << "    \"a_outer_m\": " << A_OUTER << ",\n";
    f << "    \"N_a\": " << N_A << ",\n";
    f << "    \"b_inner_m\": " << B_INNER << ",\n";
    f << "    \"b_outer_m\": " << B_OUTER << ",\n";
    f << "    \"N_b\": " << N_B << ",\n";
    f << "    \"N_phi\": " << N_PHI << ",\n";
    f << "    \"x0_unit\": \"mm\",\n";
    f << "    \"d_unit\": \"mm\",\n";
    f << "    \"M_unit\": \"H\"\n";
    f << "  },\n";

    f << "  \"x0\": [";
    for (std::size_t i = 0; i < Nx; ++i) {
        if (i) f << ", ";
        f << x0_mm[i];
    }
    f << "],\n";

    f << "  \"d\": [";
    for (std::size_t j = 0; j < Nd; ++j) {
        if (j) f << ", ";
        f << d_mm[j];
    }
    f << "],\n";

    f << std::scientific << std::setprecision(10);
    f << "  \"M\": [\n";
    for (std::size_t i = 0; i < Nx; ++i) {
        f << "    [";
        for (std::size_t j = 0; j < Nd; ++j) {
            if (j) f << ", ";
            f << M[i][j];
        }
        f << "]";
        if (i + 1 < Nx) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";

    f.close();
    std::cout << "Saved to simulation_data.json" << std::endl;

    // ============================================================
    //  第二阶段：三线圈耦合 —— σ × d 网格，x0 固定 = X0_FIXED_METAL
    // ============================================================
    constexpr int N_SIGMA = 80;
    std::vector<double> sigma_v(N_SIGMA);
    for (int i = 0; i < N_SIGMA; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(N_SIGMA - 1);
        sigma_v[i] = std::pow(10.0, 4.0 + 4.0 * t);   // 1e4 .. 1e8 S/m, 对数分布
        // 下限扩到 1e4 是为了覆盖碳纤维等低电导率材料
    }

    std::vector<std::vector<double>> M_metal(N_SIGMA, std::vector<double>(Nd, 0.0));

    std::cout << "\nComputing three-coil M_eff over "
              << N_SIGMA << " (sigma) x " << Nd << " (d) grid..." << std::endl;

    // 优化：M12/M1m/M2m/Lm 只与几何相关，与 σ 无关 —— 每个 d 只算一次
    const double r_wire = 1.0e-3;
    const double Lm     = MU0 * R_PLATE * (std::log(8.0 * R_PLATE / r_wire) - 2.0);
    const double wL     = OMEGA * Lm;

    for (std::size_t j = 0; j < Nd; ++j) {
        const double h_axial = d_mm[j] * 1.0e-3;
        const double z_plate = 0.5 * h_axial;             // 金属板放在两线圈中点
        const double M12 = calc_M_total(A_INNER, A_OUTER, N_A,
                                        B_INNER, B_OUTER, N_B,
                                        h_axial, X0_FIXED_METAL);
        const double M1m = calc_M_total(A_INNER, A_OUTER, N_A,
                                        R_PLATE, R_PLATE, 1,
                                        z_plate, 0.0);
        const double M2m = calc_M_total(R_PLATE, R_PLATE, 1,
                                        B_INNER, B_OUTER, N_B,
                                        h_axial - z_plate, X0_FIXED_METAL);
        for (int i = 0; i < N_SIGMA; ++i) {
            const double sigma = sigma_v[i];
            const double Rm    = 2.0 * PI / (sigma * T_PLATE);
            const double D     = Rm * Rm + wL * wL;
            const double Re    = M12 - OMEGA * OMEGA * M1m * M2m * Lm / D;
            const double Im    = -OMEGA * M1m * M2m * Rm / D;
            M_metal[i][j] = std::sqrt(Re * Re + Im * Im);
        }
    }
    std::cout << "  done." << std::endl;

    std::ofstream g("simulation_data_metal.json");
    if (!g) {
        std::cerr << "Failed to open simulation_data_metal.json." << std::endl;
        return 1;
    }
    g << "{\n";
    g << "  \"parameters\": {\n";
    g << "    \"method\": \"three-coil coupling (TX-metal-RX) via elliptic integrals\",\n";
    g << "    \"frequency_Hz\": " << F_OP << ",\n";
    g << "    \"plate_radius_m\": " << R_PLATE << ",\n";
    g << "    \"plate_thickness_m\": " << T_PLATE << ",\n";
    g << "    \"x0_fixed_m\": " << X0_FIXED_METAL << ",\n";
    g << "    \"a_inner_m\": " << A_INNER << ",\n";
    g << "    \"a_outer_m\": " << A_OUTER << ",\n";
    g << "    \"b_inner_m\": " << B_INNER << ",\n";
    g << "    \"b_outer_m\": " << B_OUTER << ",\n";
    g << "    \"sigma_unit\": \"S/m\",\n";
    g << "    \"d_unit\": \"mm\",\n";
    g << "    \"M_unit\": \"H\"\n";
    g << "  },\n";

    g << std::scientific << std::setprecision(6);
    g << "  \"sigma\": [";
    for (int i = 0; i < N_SIGMA; ++i) { if (i) g << ", "; g << sigma_v[i]; }
    g << "],\n";

    g.unsetf(std::ios::scientific);
    g << "  \"d\": [";
    for (std::size_t j = 0; j < Nd; ++j) { if (j) g << ", "; g << d_mm[j]; }
    g << "],\n";

    g << std::scientific << std::setprecision(10);
    g << "  \"M\": [\n";
    for (int i = 0; i < N_SIGMA; ++i) {
        g << "    [";
        for (std::size_t j = 0; j < Nd; ++j) {
            if (j) g << ", ";
            g << M_metal[i][j];
        }
        g << "]";
        if (i + 1 < N_SIGMA) g << ",";
        g << "\n";
    }
    g << "  ]\n";
    g << "}\n";
    g.close();
    std::cout << "Saved to simulation_data_metal.json" << std::endl;

    return 0;
}
