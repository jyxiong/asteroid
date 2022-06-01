#include "Common/Object/hittable.h"

class translate : public hittable {
public:
    translate(std::shared_ptr<hittable> object_ptr, const vec3 &offset)
        : m_object_ptr(std::move(object_ptr)), m_offset(offset) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;
public:
    std::shared_ptr<hittable> m_object_ptr;
    vec3 m_offset;
};

bool translate::hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
    ray moved_r(r.origin() - m_offset, r.direction(), r.time());
    if (!m_object_ptr->hit(moved_r, t_min, t_max, rec))
        return false;

    rec.p += m_offset;
    rec.set_face_normal(moved_r, rec.normal);
    return true;
}

bool translate::bounding_box(double time0, double time1, aabb &output_box) const {
    if (!m_object_ptr->bounding_box(time0, time1, output_box))
        return false;

    output_box = aabb(output_box.min() + m_offset, output_box.max() + m_offset);
    return true;
}
