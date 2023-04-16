#include <cstdio>

#include <cuda_runtime.h>

class Object
{
public:
    int my_id;
};

class Sphere : public Object
{
};

class Scene
{
public:
    Object **objects; // pointer to a objects of pointers to hittable
};

const int length = 5;

__global__ void test_kernel(Scene *scene)
{
    for (int i = 0; i < length; i++)
        printf("object: %d, id: %d\n", i, scene->objects[i]->my_id);
}

int main()
{
    Scene h_scene{};
    h_scene.objects = new Object *[length];
    for (int i = 0; i < length; ++i)
    {
        h_scene.objects[i] = new Sphere();
        h_scene.objects[i]->my_id = i + 1; // so we can prove that things are working
    }

    // scene_ptr → scene
    //                  .objects → [ptr_object] → object
    //                             [ptr_object] → object
    //                             ......
    //                             [ptr_object] → object

    // 将scene拷贝到GPU
    Scene *d_scene;
    cudaMalloc(&d_scene, sizeof(Scene));
    cudaMemcpy(d_scene, &h_scene, sizeof(Scene), cudaMemcpyHostToDevice);

    // 将每个object拷贝到GPU
    auto **d_objects = new Object *[length];
    for (int i = 0; i < length; ++i)
    {
        cudaMalloc(&d_objects[i], sizeof(Sphere));
        printf("\tCopying data\n");
        cudaMemcpy(d_objects[i], h_scene.objects[i], sizeof(Sphere), cudaMemcpyHostToDevice);
    }

    // 将数组的值拷贝到GPU，也就是每个object的指针
    Object **d_ptr_objects;
    cudaMalloc(&d_ptr_objects, length * sizeof(Object *));
    cudaMemcpy(d_ptr_objects, d_objects, length * sizeof(Object *), cudaMemcpyHostToDevice);

    // 将数组的指针拷贝到GPU，同时也是scene的成员变量
    cudaMemcpy(&(d_scene->objects), &(d_ptr_objects), sizeof(Object **), cudaMemcpyHostToDevice);

    test_kernel<<<1, 1>>>(d_scene);
    cudaDeviceSynchronize();

    return 0;
}