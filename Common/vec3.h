#pragma once

#include <cmath>
#include <ostream>

class vec3 {
public:
    vec3() : m_element{0, 0, 0} {}
    vec3(double e0, double e1, double e2) : m_element{e0, e1, e2} {}

    double x() const { return m_element[0]; }
    double y() const { return m_element[1]; }
    double z() const { return m_element[2]; }

    // operator-
    vec3 operator-() const { return vec3{-m_element[0], -m_element[1], -m_element[2]}; }

    // operator[]
    double operator[](int i) const { return m_element[i]; }
    double &operator[](int i) { return m_element[i]; };

    // operator+=
    vec3 &operator+=(const vec3 &v) {
        m_element[0] += v[0];
        m_element[1] += v[1];
        m_element[2] += v[2];
        return *this;
    }

    // operator*=
    vec3 &operator*=(double t) {
        m_element[0] += t;
        m_element[1] += t;
        m_element[2] += t;
        return *this;
    }

    // operator/=
    vec3 &operator/=(double t) {
        return *this *= 1 / t;
    }

    // length
    double length() {
        return sqrt(length_squared());
    }

    // length_squared
    double length_squared() {
        return m_element[0] * m_element[0] + m_element[1] * m_element[1] + m_element[2] * m_element[2];
    }

    // near zero
    bool near_zero() {
        const auto s = 1e-8;
        return (fabs(m_element[0]) < s) && (fabs(m_element[1]) < s) && (fabs(m_element[2]) < s);
    }

public:
    double m_element[3];
};

// type alias
using color = vec3;
using point3 = vec3;

// vec3 utility function
// operator <<
inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

// operator +
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3{u.x() + v.x(), u.y() + v.y(), u.z() + v.z()};
}

// operator +
inline vec3 operator+(const vec3 &v, double t) {
    return vec3{v.x() + t, v.y() + t, v.z() + t};
}

// operator +
inline vec3 operator+(double t, const vec3 &v) {
    return v + t;
}

// operator -
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3{u.x() - v.x(), u.y() - v.y(), u.z() - v.z()};
}

// operator *
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3{u.x() * v.x(), u.y() * v.y(), u.z() * v.z()};
}

// operator *
inline vec3 operator*(const vec3 &v, double t) {
    return vec3{v.x() * t, v.y() * t, v.z() * t};
}

// operator *
inline vec3 operator*(double t, const vec3 &v) {
    return v * t;
}

// operator /
inline vec3 operator/(const vec3 &v, double t) {
    return 1 / t * v;
}

// dot
inline double dot(const vec3 &u, const vec3 &v) {
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

// cross
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3{u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x()};
}

// unit vector
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

// reflect vector
vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(n, v) * n;
}

// refract vector
vec3 refract(const vec3 &v, const vec3 &n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-v, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}