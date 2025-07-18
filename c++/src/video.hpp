#include <opencv2/opencv.hpp>
#if _WIN32
#include <opencv2/objdetect/aruco_detector.hpp>
#else
#include <opencv2/aruco.hpp>
#endif
#include <iostream>

constexpr auto WIDTH_1 = 216.2f;
constexpr auto WIDTH_2 = 216.1f;
constexpr auto WIDTH_3 = 101.5f;
constexpr auto WIDTH_4 = 104.1f;
constexpr auto HEIGHT_1 = 114.1f;
constexpr auto HEIGHT_2 = 113.1f;
constexpr auto HEIGHT_3 = 50.4f;
constexpr auto HEIGHT_4 = 56.7f;
constexpr auto PI = 3.141592654f;

std::vector<cv::Point2f> SRC_LANDMARKS = {
    cv::Point2f(0, 0),
    cv::Point2f(WIDTH_3, 0),
    cv::Point2f(WIDTH_1, 0),
    cv::Point2f((1 - HEIGHT_4 / HEIGHT_2) * WIDTH_1 + HEIGHT_4 / HEIGHT_2 * WIDTH_2, HEIGHT_4),
    cv::Point2f((WIDTH_1 + WIDTH_2) / 2, HEIGHT_2),
    cv::Point2f(WIDTH_4, ((1 - WIDTH_4 / WIDTH_2) * HEIGHT_1 + WIDTH_4 / WIDTH_2 * HEIGHT_2)),
    cv::Point2f(0, HEIGHT_1),
    cv::Point2f(0, HEIGHT_3),
};

cv::Mat debugImage;
bool hasDebug;
void debug(cv::Mat im) {
    im.copyTo(debugImage);
    hasDebug = true;
}

struct Options {
    int thresholdValue = 65;
    int sharpen = 6;
    float pointLimit = 6;
};

struct Candidate {
    cv::Point screenPoint;
    std::vector<cv::Point2f> hitPositions;
    int hitTimes;
    int firstSeeen;
    int lastSeen;
};

struct Transform {
    std::vector<double> coeffs;
    bool initialized = false;
    double cosa;
    double sina;

    Transform() {
    }

    Transform(double rotation, const std::vector<double>& coeffs, bool initialized) {
        cosa = cos(rotation);
        sina = sin(rotation);
        this->coeffs = coeffs;
        this->initialized = initialized;
    }

    cv::Point2f findInverse(const cv::Point2f& pt) {
        double A = coeffs[0], B = coeffs[1], C = coeffs[2], X0 = coeffs[3], Y0 = coeffs[4], x0 = coeffs[5], y0 = coeffs[6];        
        double x = pt.x * cosa + pt.y * sina;
        double y = pt.y * cosa - pt.x * sina;
        double dx = 1.0f / (x - x0);
        double dy = 1.0f / (y - y0);
        double Dx = dx * X0;
        double Dy = dy * Y0;
        double X = (B * Dx - B * Dy - C * dy - Dx * dy) / (A * dy + B * dx - dx * dy);
        double Y = (-A * Dx + A * Dy - C * dx - Dy * dx) / (A * dy + B * dx - dx * dy);
        return cv::Point2f((float)X, (float)Y);
    }

};

// Checks if list has point within distance limit
int hasPoint(const std::vector<cv::Point>& points, const cv::Point& point, float limit = 3) {
    for (int i = 0; i < points.size(); i++) {
        int dx = points[i].x - point.x;
        int dy = points[i].y - point.y;
        if (dx * dx + dy * dy < limit * limit)
            return i;
    }
    return -1;
}

int hasPoint(const std::vector<Candidate>& candidates, cv::Point point, float limit = 3) {
    std::vector<cv::Point> points;
    for (const auto& candidate : candidates) {
        points.push_back(candidate.screenPoint);
    }
    return hasPoint(points, point, limit);
}

std::tuple<double, double> lingress(const std::vector<double>& X, const std::vector<double>& y) {
    double Sxx = 0, Sxy = 0, Sx = 0, Sy = 0;
    int n = (int) X.size();

    for (int i = 0; i < n; i++) {
        Sxx += X[i] * X[i];
        Sxy += X[i] * y[i];
        Sx += X[i];
        Sy += y[i];
    }

    double a = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx);
    double b = (Sy * Sxx - Sx * Sxy) / (n * Sxx - Sx * Sx);
    return std::make_tuple(a, b);
}

double findRotationAngle(const std::vector<cv::Point2f>& topPoints, const std::vector<cv::Point2f>& bottomPoints){

    std::vector<double> X;
    std::vector<double> y;
    for (auto& pt : topPoints) {
        X.push_back(pt.x);
        y.push_back(pt.y);
    }

    double a1 = std::get<0>(lingress(X, y));
    
    X.clear();
    y.clear();

    for (auto& pt : bottomPoints) {
        X.push_back(pt.x);
        y.push_back(pt.y);
    }

    double a2 = std::get<0>(lingress(X, y));
    return atan((a1 + a2) / 2);
}

double findRotationAngle(const std::vector<cv::Point2f>& landMarks) {
    std::vector<cv::Point2f> topPoints = {
        landMarks[0], landMarks[1], landMarks[2]
    };
    std::vector<cv::Point2f> bottomPoints = {
        landMarks[6], landMarks[5], landMarks[4]
    };
    return findRotationAngle(topPoints, bottomPoints);
}

std::vector<double> linSolve4x4(const std::vector<double>& A, const std::vector<double>& b) {
    double a11 = A[0], a12 = A[1], a13 = A[2], a14 = A[3];
    double a21 = A[4], a22 = A[5], a23 = A[6], a24 = A[7];
    double a31 = A[8], a32 = A[9], a33 = A[10], a34 = A[11];
    double a41 = A[12], a42 = A[13], a43 = A[14], a44 = A[15];
    double det = (
        a11 * a22 * a33 * a44 
        - a11 * a22 * a34 * a43 
        - a11 * a23 * a32 * a44 
        + a11 * a23 * a34 * a42 
        + a11 * a24 * a32 * a43 
        - a11 * a24 * a33 * a42 
        - a12 * a21 * a33 * a44 
        + a12 * a21 * a34 * a43 
        + a12 * a23 * a31 * a44 
        - a12 * a23 * a34 * a41 
        - a12 * a24 * a31 * a43 
        + a12 * a24 * a33 * a41 
        + a13 * a21 * a32 * a44 
        - a13 * a21 * a34 * a42 
        - a13 * a22 * a31 * a44 
        + a13 * a22 * a34 * a41 
        + a13 * a24 * a31 * a42 
        - a13 * a24 * a32 * a41 
        - a14 * a21 * a32 * a43 
        + a14 * a21 * a33 * a42 
        + a14 * a22 * a31 * a43 
        - a14 * a22 * a33 * a41 
        - a14 * a23 * a31 * a42 
        + a14 * a23 * a32 * a41
    );

    double i11 = (a22 * a33 * a44 - a22 * a34 * a43 - a23 * a32 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a24 * a33 * a42);
    double i12 = (-a12 * a33 * a44 + a12 * a34 * a43 + a13 * a32 * a44 - a13 * a34 * a42 - a14 * a32 * a43 + a14 * a33 * a42);
    double i13 = (a12 * a23 * a44 - a12 * a24 * a43 - a13 * a22 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a14 * a23 * a42);
    double i14 = (-a12 * a23 * a34 + a12 * a24 * a33 + a13 * a22 * a34 - a13 * a24 * a32 - a14 * a22 * a33 + a14 * a23 * a32);
    double i21 = (-a21 * a33 * a44 + a21 * a34 * a43 + a23 * a31 * a44 - a23 * a34 * a41 - a24 * a31 * a43 + a24 * a33 * a41);
    double i22 = (a11 * a33 * a44 - a11 * a34 * a43 - a13 * a31 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a14 * a33 * a41);
    double i23 = (-a11 * a23 * a44 + a11 * a24 * a43 + a13 * a21 * a44 - a13 * a24 * a41 - a14 * a21 * a43 + a14 * a23 * a41);
    double i24 = (a11 * a23 * a34 - a11 * a24 * a33 - a13 * a21 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a14 * a23 * a31);
    double i31 = (a21 * a32 * a44 - a21 * a34 * a42 - a22 * a31 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a24 * a32 * a41);
    double i32 = (-a11 * a32 * a44 + a11 * a34 * a42 + a12 * a31 * a44 - a12 * a34 * a41 - a14 * a31 * a42 + a14 * a32 * a41);
    double i33 = (a11 * a22 * a44 - a11 * a24 * a42 - a12 * a21 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a14 * a22 * a41);
    double i34 = (-a11 * a22 * a34 + a11 * a24 * a32 + a12 * a21 * a34 - a12 * a24 * a31 - a14 * a21 * a32 + a14 * a22 * a31);
    double i41 = (-a21 * a32 * a43 + a21 * a33 * a42 + a22 * a31 * a43 - a22 * a33 * a41 - a23 * a31 * a42 + a23 * a32 * a41);
    double i42 = (a11 * a32 * a43 - a11 * a33 * a42 - a12 * a31 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a13 * a32 * a41);
    double i43 = (-a11 * a22 * a43 + a11 * a23 * a42 + a12 * a21 * a43 - a12 * a23 * a41 - a13 * a21 * a42 + a13 * a22 * a41);
    double i44 = (a11 * a22 * a33 - a11 * a23 * a32 - a12 * a21 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a13 * a22 * a31);

    double b1 = b[0], b2 = b[1], b3 = b[2], b4 = b[3];

    return std::vector<double>{
        (i11* b1 + i12 * b2 + i13 * b3 + i14 * b4) / det,
        (i21* b1 + i22 * b2 + i23 * b3 + i24 * b4) / det,
        (i31* b1 + i32 * b2 + i33 * b3 + i34 * b4) / det,
        (i41* b1 + i42 * b2 + i43 * b3 + i44 * b4) / det
    };
}

std::vector<double> linSolve3x3(const std::vector<double>& A, const std::vector<double>& b) {
    double a11 = A[0], a12 = A[1], a13 = A[2];
    double a21 = A[3], a22 = A[4], a23 = A[5];
    double a31 = A[6], a32 = A[7], a33 = A[8];

    double det = (
        a11 * a22 * a33 
        - a11 * a23 * a32 
        - a12 * a21 * a33 
        + a12 * a23 * a31 
        + a13 * a21 * a32 
        - a13 * a22 * a31
    );

    double i11 = (a22 * a33 - a23 * a32) ;
    double i12 = (-a12 * a33 + a13 * a32) ;
    double i13 = (a12 * a23 - a13 * a22) ;
    double i21 = (-a21 * a33 + a23 * a31) ;
    double i22 = (a11 * a33 - a13 * a31) ;
    double i23 = (-a11 * a23 + a13 * a21) ;
    double i31 = (a21 * a32 - a22 * a31) ;
    double i32 = (-a11 * a32 + a12 * a31) ;
    double i33 = (a11 * a22 - a12 * a21) ;

    double b1 = b[0], b2 = b[1], b3 = b[2];
    return std::vector<double>{
        (i11* b1 + i12 * b2 + i13 * b3) / det,
        (i21* b1 + i22 * b2 + i23 * b3) / det,
        (i31* b1 + i32 * b2 + i33 * b3) / det
    };
}

std::vector<double> transpose(const std::vector<double>& A, int m, int n) {
    std::vector<double> res;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            res.push_back(A[int(i * n + j)]);
        }
    }
    return res;
}

// (m,n) @ (n,p) --> m,p
std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B, int m, int n, int p) {
    std::vector<double> res;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[int(i * n + k)] * B[int(k * p + j)];
            }
            res.push_back(sum);
        }
    }

    return res;
}

std::vector<double> operator +(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> res(v1.size());
    for (int i = 0; i < (int)v1.size(); i++) {
        res[i] = v1[i] + v2[i];
    }
    return res;
}

std::vector<double> operator -(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> res(v1.size());
    for (int i = 0; i < (int)v1.size(); i++) {
        res[i] = v1[i] - v2[i];
    }
    return res;
}

std::vector<double> operator *(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> res(v1.size());
    for (int i = 0; i < (int)v1.size(); i++) {
        res[i] = v1[i] * v2[i];
    }
    return res;
}

std::vector<double> square(const std::vector<double>& v) {
    std::vector<double> res(v.size());
    for (int i = 0; i < (int)v.size(); i++) {
        res[i] = v[i] * v[i];
    }
    return res;
}

std::vector<double> operator /(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> res(v1.size());
    for (int i = 0; i < (int)v1.size(); i++) {
        res[i] = v1[i] / v2[i];
    }
    return res;
}

std::vector<double> operator -(const std::vector<double>& v, double c) {
    std::vector<double> res(v.size());
    for (int i = 0; i < (int)v.size(); i++) {
        res[i] = v[i] - c;
    }
    return res;
}

std::vector<double> operator *(double c, const std::vector<double>& v) {
    std::vector<double> res(v.size());
    for (int i = 0; i < (int)v.size(); i++) {
        res[i] = v[i] * c;
    }
    return res;
}

std::vector<double> operator /(double c, const std::vector<double>& v) {
    std::vector<double> res(v.size());
    for (int i = 0; i < (int)v.size(); i++) {
        res[i] = c / v[i];
    }
    return res;
}

double arr_sum(const std::vector<double>& arr) {
    double s = 0.0f;
    for (double x : arr) s += x;
    return s;
}

Transform findTransform(double rotation, const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    std::vector<double> X;   // source
    std::vector<double> Y;   // source
    std::vector<double> x;   // destination
    std::vector<double> y;   // destination

    int n = (int)srcPoints.size();

    std::vector<double> M;   // stack([-y, x, Y,-X]).T
    std::vector<double> N;   // x*Y - X * y

    double cosa = cos(rotation), sina = sin(rotation);

    for (int i = 0; i < n; i++) {
        X.push_back(srcPoints[i].x);
        Y.push_back(srcPoints[i].y);
        x.push_back(dstPoints[i].x * cosa + dstPoints[i].y * sina);
        y.push_back(dstPoints[i].y * cosa - dstPoints[i].x * sina);
        
        // [-y, x, Y,-X]
        M.push_back(-y.back());
        M.push_back(x.back());
        M.push_back(Y.back());
        M.push_back(-X.back());

        // x*Y - X * y
        N.push_back(x.back() * Y.back() - X.back() * y.back());
    }

    std::vector<double> MT = transpose(M, n, 4);
    std::vector<double> MT_M = matmul(MT, M, 4, n, 4);
    std::vector<double> MT_N = matmul(MT, N, 4, n, 1);
    std::vector<double> XYxy = linSolve4x4(MT_M, MT_N);
    double X0 = XYxy[0], Y0 = XYxy[1], x0 = XYxy[2], y0 = XYxy[3];
    double Xm = X0, Ym = Y0, xm = x0, ym = y0;

    auto ones = std::vector<double>(n, 1.0);
    std::vector<double> abcd = linSolve4x4(MT_M, matmul(MT, ones, 4, n, 1));
    double a = abcd[0], b = abcd[1], c = abcd[2], d = abcd[3];

    double eps = 1e-3f;
    for (int step = 0; step < 10; step++)
    {
        double diff = xm * Ym - Xm * ym;
        double eX = Xm - X0 - a * diff;
        double eY = Ym - Y0 - b * diff;
        double ex = xm - x0 - c * diff;
        double ey = ym - y0 - d * diff;
        double err = fabs(eX) + fabs(eY) + fabs(ex) + fabs(ey);
        //std::cout << "step :" << step << ", error:" << err << "\n";
        if (err < eps) {
            break;
        }
        std::vector<double> H = {
            1 + a * ym, -a * xm, -a * Ym, a* Xm,
            b* ym, 1 - b * xm, -b * Ym, b* Xm,
            c* ym, -c * xm, 1 - c * Ym, c* Xm,
            d* ym, -d * xm, -d * Ym, 1 + d * Xm
        };

        std::vector<double> G = { eX,eY,ex,ey };
        std::vector<double> delta = linSolve4x4(H, G);

        Xm -= delta[0];
        Ym -= delta[1];
        xm -= delta[2];
        ym -= delta[3];
    }
    X0 = Xm;
    Y0 = Ym;
    x0 = xm;
    y0 = ym;

    M.clear();
    N.clear();
    for (int i = 0; i < n; i++) {
        M.push_back(X[i] * (x[i] - x0));
        M.push_back(Y[i] * (x[i] - x0));
        M.push_back(x[i] - x0);
        N.push_back(X[i] - X0);
    }
    MT = transpose(M, n, 3);
    MT_M = matmul(MT, M, 3, n, 3);
    MT_N = matmul(MT, N, 3, n, 1);
    std::vector<double> ABC = linSolve3x3(MT_M, MT_N);
    double A = ABC[0], B = ABC[1], C = ABC[2];

    std::vector<double> dx = 1.0 / (x - x0);
    std::vector<double> dy = 1.0 / (y - y0);
    std::vector<double> Dx = X0 * dx;
    std::vector<double> Dy = Y0 * dy;

    eps = 1e-2f;

    for (int step = 0; step < 1000; step++) {
        auto Dn = (A * dy + B * dx - dx * dy);
        auto Nx = (B * Dx - B * Dy - C * dy - Dx * dy);
        auto Ny = (-A * Dx + A * Dy - C * dx - Dy * dx);
        auto Xm = Nx / Dn;
        auto Ym = Ny / Dn;
        auto Ex = Xm - X;
        auto Ey = Ym - Y;
        auto Dn2 = square(Dn);
        auto dx2 = square(dx);
        auto dy2 = square(dy);
        auto Xm2 = square(Xm);
        auto Ym2 = square(Ym);

        double D_A = arr_sum(-2 * dy * Ex * Nx / Dn2 + Ey * (-2 * dy * Ny / Dn2 + 2 * (Dy - Dx) / Dn));
        double D_B = arr_sum(-2 * dx * Ey * Ny / Dn2 + Ex * (-2 * dx * Nx / Dn2 + 2 * (Dx - Dy) / Dn));
        double D_C = arr_sum(-2 * dx * Ey / Dn - 2 * dy * Ex / Dn);
        double D_AA = arr_sum(2 * (2 * dy2 * Ex * Xm + dy2 * Xm2 - 2 * dy * Ey * (Dy - Dx - dy * Ym) + square(Dy - Dx - dy * Ym)) / Dn2);
        double D_AB = arr_sum(2 * (2 * dx * dy * Ex * Xm - dx * Ey * (Dy - Dx - 2 * dy * Ym) - dx * (Dy - Dx - dy * Ym) * Ym - dy * Ex * (Dx - Dy) - dy * (Dx - Dy - dx * Xm) * Xm) / Dn2);
        double D_AC = arr_sum(2 * (dx * dy * Ey - dx * (Dy - Dx - dy * Ym) + dy2 * Ex + dy2 * Xm) / Dn2);
        double D_BB = arr_sum(2 * (2 * dx2 * Ey * Ym + dx2 * Ym2 - 2 * dx * Ex * (Dx - Dy - dx * Xm) + square(Dx - Dy - dx * Xm)) / Dn2);
        double D_BC = arr_sum(2 * (dx2 * Ey + dx2 * Ym + dx * dy * Ex - dy * (Dx - Dy - dx * Xm)) / Dn2);
        double D_CC = arr_sum(2 * (dx2 + dy2) / Dn2);

        if (fabs(D_A) + fabs(D_B) + fabs(D_C) < eps) {
            break;
        }

        std::vector<double> G = {
            D_A, D_B, D_C
        };

        std::vector<double> H = {
            D_AA, D_AB, D_AC,
            D_AB, D_BB, D_BC,
            D_AC, D_BC, D_CC
        };

        std::vector<double> delta = linSolve3x3(H, G);
        A -= delta[0];
        B -= delta[1];
        C -= delta[2];
    }

    auto coeffs = std::vector<double>{
        A, B, C, X0, Y0, x0, y0
    };
    return Transform(rotation, coeffs, true);
}

cv::Point2f getCenter(const std::vector<cv::Point2f>& corner) {
    float sx = 0;
    float sy = 0;
    for (auto& pt : corner) {
        sx += pt.x;
        sy += pt.y;
    }
    return cv::Point2f(sx / corner.size(), sy / corner.size());
}

cv::Point2f getTopLeft(const std::vector<cv::Point2f>& corner) {
    cv::Point2f res = corner[0];
    for (auto& pt : corner) {
        if (pt.x + pt.y < res.x + res.y) {
            res = pt;
        }
    }
    return res;
}

cv::Point2f getTopRight(const std::vector<cv::Point2f>& corner) {
    cv::Point2f res = corner[0];
    for (auto& pt : corner) {
        if (pt.x - pt.y > res.x - res.y) {
            res = pt;
        }
    }
    return res;
}

cv::Point2f getBottomLeft(const std::vector<cv::Point2f>& corner) {
    cv::Point2f res = corner[0];
    for (auto& pt : corner) {
        if (pt.x - pt.y < res.x - res.y) {
            res = pt;
        }
    }
    return res;
}

cv::Point2f getBottomRight(const std::vector<cv::Point2f>& corner) {
    cv::Point2f res = corner[0];
    for (auto& pt : corner) {
        if (pt.x + pt.y > res.x + res.y) {
            res = pt;
        }
    }
    return res;
}

std::vector<cv::Point2f> findLandMarks(const std::vector<std::vector<cv::Point2f>>& corners) {
    float ymin = 1e6;
    float ymax = 0;
    for (auto& corner : corners) {
        for (auto& pt : corner) {
            if (pt.y < ymin) ymin = pt.y;
            if (pt.y > ymax) ymax = pt.y;
        }
    }
    float y25 = (ymin * 3 + ymax) / 4;
    float y75 = (ymin + 3 * ymax) / 4;
    std::vector< std::vector<cv::Point2f>> topCorners;
    std::vector< std::vector<cv::Point2f>> midCorners;
    std::vector< std::vector<cv::Point2f>> bottomCorners;
    
    for (auto& corner : corners) {
        float yc = getCenter(corner).y;
        if (yc < y25) {
            topCorners.push_back(corner);
        }
        else if (yc < y75) {
            midCorners.push_back(corner);
        }
        else {
            bottomCorners.push_back(corner);
        }
    }

    auto orderByCenterX = [](const std::vector<cv::Point2f>& lhs, const std::vector<cv::Point2f>& rhs) {
        return getCenter(lhs).x < getCenter(rhs).x;
    };

    std::sort(topCorners.begin(), topCorners.end(), orderByCenterX);
    std::sort(midCorners.begin(), midCorners.end(), orderByCenterX);
    std::sort(bottomCorners.begin(), bottomCorners.end(), orderByCenterX);

    return std::vector<cv::Point2f>{
        getTopLeft(topCorners[0]),
        getTopLeft(topCorners[1]),
        getTopRight(topCorners[2]),
        getTopRight(midCorners[1]),
        getBottomRight(bottomCorners[2]),
        getBottomLeft(bottomCorners[1]),
        getBottomLeft(bottomCorners[0]),
        getTopLeft(midCorners[0])
    };
}

void processVideo(cv::VideoCapture cap, std::function<void(int, float, float, int, float)> onHitCallback, std::function<bool(cv::Mat)> drawCallback, const Options& options = Options());

void processVideo(const char* filename, std::function<void(int, float, float, int, float)> onHitCallback = nullptr, std::function<bool(cv::Mat)> drawCallback = nullptr, const Options& options = Options()) {
    std::cout << "Opening file " << filename << "\n";
    cv::VideoCapture cap(filename);
    processVideo(cap, onHitCallback, drawCallback, options);
}

void processVideo(int id, std::function<void(int, float, float, int, float)> onHitCallback = nullptr, std::function<bool(cv::Mat)> drawCallback = nullptr, const Options& options = Options()) {
    std::cout << "Opening video device " << id << "\n";
    cv::VideoCapture cap(id);
    processVideo(cap, onHitCallback, drawCallback, options);
}

void processVideo(cv::VideoCapture cap, std::function<void(int, float, float, int, float)> onHitCallback, std::function<bool(cv::Mat)> drawCallback, const Options& options) {
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file!" << std::endl;
        return;// 1;
    }

    std::cout << "Threshold Value: " << options.thresholdValue << std::endl;
    std::cout << "Sharpen: " << options.sharpen << std::endl;
    std::cout << "Point limit: " << options.pointLimit << std::endl;

#if _WIN32
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::aruco::ArucoDetector detector = cv::aruco::ArucoDetector();
    detector.setDictionary(dictionary);
#else
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
#endif

    // Used to exclude corner tafs form hits
    std::vector<std::vector<cv::Point2f>> corners;

    // Candidates are possible detections that have not been confirmed in subsequent frames yet
    std::vector<Candidate> candidates;

    // Confirmed hits
    std::vector<cv::Point> hits;
    std::vector<int> hitTimes;

    // Used to avoid false positives if it takes a couple of frames to find the corners
    bool hasFoundCorners = false;
    float fps = (float) cap.get(cv::CAP_PROP_FPS);
    
    int hitNumber = 0;
    int time = -1;
    cv::Mat frame, gray, binary, lastBinary, tmp;
    std::vector<cv::Point2f> landMarks, oldLandMarks;

    while (cap.read(frame)) {
        time += 1;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Only detect corners until they are found. Assume they don't move.
        if(!hasFoundCorners) {
            std::vector<int> ids;

#if _WIN32
            detector.detectMarkers(gray, corners, ids);
#else
            cv::aruco::detectMarkers(gray, dictionary, corners, ids);
#endif
            hasFoundCorners = ids.size() == 8;
        }

        // Wide soft sharpen to even out gradual differences across the canvas
        cv::blur(gray, tmp, cv::Size(50, 50));
        cv::addWeighted(gray, 1.48, tmp, -.52, 0, gray);

        // Small sharper sharpen to emphasise dots
        cv::blur(gray, tmp, cv::Size(options.sharpen, options.sharpen));
        cv::addWeighted(gray, 1.7, tmp, -.57, 0, gray);

        // Apply threshold
        cv::threshold(gray, binary, options.thresholdValue, 255, cv::THRESH_BINARY_INV);
        // debug(binary);

        // Find countours of objects
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        if (corners.size() == 8) {
            landMarks = findLandMarks(corners);
        }
        else {
            landMarks = oldLandMarks;
        }

        oldLandMarks = landMarks;

        Transform trans1;
        Transform trans2;

        std::vector<cv::Point> targetArea = {
            landMarks[0], landMarks[2], landMarks[4], landMarks[6]
        };

        // Clean out stale candidate hits if they do not get confirmed n subsequent frames
        for(int i = (int) candidates.size() - 1; i >= 0; i--) {
            if(candidates[i].lastSeen >= time - 15)
                continue;

            candidates.erase(candidates.begin() + i);
        }

        // Loop through all the fund contours
        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            cv::Rect boundingRect = cv::boundingRect(contour);

            // Find hits
            if (area < 100 && hasFoundCorners) {
                // Find the center of the square
                cv::Point2f center = (boundingRect.br() + boundingRect.tl()) * 0.5f;

                bool cornerHit = false;
                // Exclude the 4 corner areas form target area
                for(const auto& corner : corners) {
                    if(cv::pointPolygonTest(corner, center, true) > -10)
                        cornerHit = true;
                }
                
                if(cv::pointPolygonTest(targetArea, center, false) > 0 && !cornerHit) {
                    int idx = hasPoint(candidates, center, options.pointLimit);
                    cv::Point2f srcCenter;
                    if ((center.x - landMarks[0].x) / (landMarks[2].x - landMarks[0].x) < WIDTH_3 / WIDTH_1) {
                        if (!trans1.initialized) {
                            std::vector<cv::Point2f> srcPoints = {
                                SRC_LANDMARKS[0],
                                SRC_LANDMARKS[1],
                                SRC_LANDMARKS[5],
                                SRC_LANDMARKS[6],
                                SRC_LANDMARKS[7],
                            };
                            std::vector<cv::Point2f> dstPoints = {
                                landMarks[0],
                                landMarks[1],
                                landMarks[5],
                                landMarks[6],
                                landMarks[7],
                            };
                            double rotation = findRotationAngle(landMarks);
                            trans1 = findTransform(rotation, srcPoints, dstPoints);
                        }
                        srcCenter = trans1.findInverse(center);
                    }
                    else {
                        if (!trans2.initialized) {
                            std::vector<cv::Point2f> srcPoints = {
                                SRC_LANDMARKS[1],
                                SRC_LANDMARKS[2],
                                SRC_LANDMARKS[3],
                                SRC_LANDMARKS[4],
                                SRC_LANDMARKS[5],
                            };
                            std::vector<cv::Point2f> dstPoints = {
                                landMarks[1],
                                landMarks[2],
                                landMarks[3],
                                landMarks[4],
                                landMarks[5],
                            };
                            double rotation = findRotationAngle(landMarks);
                            trans2 = findTransform(rotation, srcPoints, dstPoints);
                        }
                        srcCenter = trans2.findInverse(center);
                    }

                    if(idx >= 0) {
                        candidates[idx].lastSeen = time;
                        candidates[idx].hitPositions.push_back(srcCenter);

                        // This is alreadya candidate
                        int framesToQualify = 5;
                        if(candidates[idx].hitTimes ++ >= framesToQualify) {
                            int firstSeen = candidates[idx].firstSeeen;
                            hits.push_back(candidates[idx].screenPoint);
                            hitTimes.push_back(firstSeen);

                            if(onHitCallback != nullptr && firstSeen >= framesToQualify) {
                                float sx = 0.0f, sy = 0.0f;
                                auto& hitPositions = candidates[idx].hitPositions;
                                int n = (int) hitPositions.size();
                                for (int i = 0; i < n; i++)
                                {
                                    sx += hitPositions[i].x;
                                    sy += hitPositions[i].y;
                                }
                                float posX = sx / n;
                                float posY = sy / n;
                                onHitCallback(++hitNumber, posX, posY, firstSeen + 1, (firstSeen + 1) / fps);
                            }

                            candidates.erase(candidates.begin() + idx);
                        }
                    }
                    else if(hasPoint(hits, center, options.pointLimit) == -1) {
                        Candidate candidate;
                        candidate.screenPoint = center;
                        candidate.hitTimes = 1;
                        candidate.lastSeen = time;
                        candidate.firstSeeen = time;
                        candidate.hitPositions.push_back(srcCenter);
                        candidates.push_back(candidate);
                    }

                }
            }
        }

        // Display the frame
        if(drawCallback != nullptr) {
            // Draw canvas rectangle
            for (size_t i = 0; i < targetArea.size(); ++i) {
                cv::line(frame, targetArea[i], targetArea[(i + 1) % targetArea.size()], cv::Scalar(0, 255, 255), 2);
            }

            // Draw corner tag rectangles
            for (auto corner : corners) {
                for (size_t i = 0; i < corner.size(); ++i) {
                    cv::line(frame, corner[i], corner[(i + 1) % corner.size()], cv::Scalar(0, 255, 255), 2);
                }
            }

            // Draw confirmed hits
            for (auto hit : hits) {
                cv::circle(frame, hit, 10, cv::Scalar(0, 0, 255), 2);
            }

            // Draw candidate hits
            for (auto candidate : candidates) {
                cv::circle(frame, candidate.screenPoint, 10, cv::Scalar(0, 255, 255), 1);
            }

            if(!drawCallback(frame))
                break;
        }
        // if(hasDebug)
        //     cv::imshow("Debug", debugImage);
    }

    cap.release();
}

// A test to list cameras
std::string listCameras(int maxTested = 10) {
    std::ostringstream oss;
    for (int i = 0; i < maxTested; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            oss << i << ", \n";
            cap.release();
        }
    }
    std::string cameraList = oss.str();
    if (cameraList.empty()) {
        cameraList = "No cameras available.\n";
    }
    return cameraList;
}