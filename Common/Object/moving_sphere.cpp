#include "moving_sphere.h"

moving_sphere::moving_sphere(const vec3 &center0,
                             const vec3 &center1,
                             double time0,
                             double time1,
                             double radius,
                             std::shared_ptr<material> mat_ptr)
    : m_center0(center0),
      m_center1(center1),
      m_time0(time0),
      m_time1(time1),
      m_radius(radius),
      m_mat_ptr(std::move(mat_ptr)) {}

bool moving_sphere::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    auto time_center = center(r.time());
    vec3 oc = r.origin() - time_center;
    auto a = r.direction().length_squared();
    auto half_b = dot(r.direction(), oc);
    auto c = oc.length_squared() - m_radius * m_radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrt_d = sqrt(discriminant);

    auto root = (-half_b - sqrt_d) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrt_d) / a;
        if (root < t_min || root > t_max)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);

    auto out_forward = (rec.p - time_center) / m_radius;
    rec.set_face_normal(r, out_forward);

    rec.mat_ptr = m_mat_ptr;
    return true;
}

bool moving_sphere::bounding_box(double time0, double time1, aabb &output_box) const {
    aabb box0(center(time0) - vec3(m_radius, m_radius, m_radius),
              center(time0) + vec3(m_radius, m_radius, m_radius));

    aabb box1(center(time1) - vec3(m_radius, m_radius, m_radius),
              center(time1) + vec3(m_radius, m_radius, m_radius));

    output_box = aabb::surrounding_box(box0, box1);
    return true;
}

vec3 moving_sphere::center(double time) const {
    return m_center0 + ((time - m_time0) / (m_time1 - m_time0)) * (m_center1 - m_center0);
}