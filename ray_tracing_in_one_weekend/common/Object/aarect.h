#pragma once

#include <memory>
#include <utility>
#include "Common/Material/material.h"
#include "Common/Object/hittable.h"
#include "Common/aabb.h"

class xy_rect : public hittable {
public:
    xy_rect() = default;
    xy_rect(double x0, double x1, double y0, double y1, double k, std::shared_ptr<material> mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;
private:
    double m_x0{}, m_x1{};
    double m_y0{}, m_y1{};
    double m_k{};
    std::shared_ptr<material> m_mat_ptr;
};

class xz_rect : public hittable {
public:
    xz_rect() = default;
    xz_rect(double x0, double x1, double z0, double z1, double k, std::shared_ptr<material> mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

public:
    std::shared_ptr<material> m_mat_ptr;
    double m_x0{}, m_x1{}, m_z0{}, m_z1{}, m_k{};
};

class yz_rect : public hittable {
public:
    yz_rect() = default;
    yz_rect(double y0, double y1, double z0, double z1, double k, std::shared_ptr<material> mat_ptr);

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool bounding_box(double time0, double time1, aabb &output_box) const override;

public:
    std::shared_ptr<material> m_mat_ptr;
    double m_y0{}, m_y1{}, m_z0{}, m_z1{}, m_k{};
};
