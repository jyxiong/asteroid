#include "aarect.h"

xy_rect::xy_rect(double x0, double x1, double y0, double y1, double k, std::shared_ptr<material> mat_ptr)
    : m_x0(x0), m_x1(x1), m_y0(y0), m_y1(y1), m_k(k), m_mat_ptr(std::move(mat_ptr)) {};

bool xy_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    auto t = (m_k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max)
        return false;

    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < m_x0 || x > m_x1 || y < m_y0 || y > m_y1)
        return false;

    rec.t = t;
    rec.p = r.at(t);

    auto outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);

    rec.u = (x - m_x0) / (m_x1 - m_x0);
    rec.v = (y - m_y0) / (m_y1 - m_y0);
    rec.mat_ptr = m_mat_ptr;

    return true;
}

bool xy_rect::bounding_box(double time0, double time1, aabb &output_box) const {
    output_box = aabb(point3(m_x0, m_y0, m_k - 0.0001), point3(m_x1, m_y1, m_k + 0.0001));
    return true;
}

xz_rect::xz_rect(double x0, double x1, double z0, double z1, double k, std::shared_ptr<material> mat_ptr)
    : m_x0(x0), m_x1(x1), m_z0(z0), m_z1(z1), m_k(k), m_mat_ptr(std::move(mat_ptr)) {};

bool xz_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    auto t = (m_k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < m_x0 || x > m_x1 || z < m_z0 || z > m_z1)
        return false;
    rec.u = (x - m_x0) / (m_x1 - m_x0);
    rec.v = (z - m_z0) / (m_z1 - m_z0);
    rec.t = t;
    auto outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = m_mat_ptr;
    rec.p = r.at(t);
    return true;
}

bool xz_rect::bounding_box(double time0, double time1, aabb &output_box) const {
    output_box = aabb(point3(m_x0, m_k - 0.0001, m_z0), point3(m_x1, m_k + 0.0001, m_z1));
    return true;
}

yz_rect::yz_rect(double y0, double y1, double z0, double z1, double k, std::shared_ptr<material> mat_ptr)
    : m_y0(y0), m_y1(y1), m_z0(z0), m_z1(z1), m_k(k), m_mat_ptr(std::move(mat_ptr)) {};

bool yz_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    auto t = (m_k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max)
        return false;
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < m_y0 || y > m_y1 || z < m_z0 || z > m_z1)
        return false;
    rec.u = (y - m_y0) / (m_y1 - m_y0);
    rec.v = (z - m_z0) / (m_z1 - m_z0);
    rec.t = t;
    auto outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = m_mat_ptr;
    rec.p = r.at(t);
    return true;
}
bool yz_rect::bounding_box(double time0, double time1, aabb &output_box) const {
    output_box = aabb(point3(m_k - 0.0001, m_y0, m_z0), point3(m_k + 0.0001, m_y1, m_z1));
    return true;
}