#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <windows.h>
#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>


using namespace std;


const int Num = 1000;
const int pasNum = 15000;
const int lieNum = 40000;
unsigned int Act[lieNum * Num] = { 0 };
unsigned int Pas[lieNum * Num] = { 0 };



void init_A()
{

    unsigned int a;
    ifstream infile("act2.txt");
    char fin[10000] = { 0 };
    int index;

    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;


        while (line >> a)
        {
            if (biaoji == 0)
            {

                index = a;
                biaoji = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Act[index * (Num + 1) + Num - 1 - j] += temp;
            Act[index * (Num + 1) + Num] = 1;
        }
    }
}

void init_P()
{
    unsigned int a;
    ifstream infile("pas2.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int biaoji = 0;

        while (line >> a)
        {
            if (biaoji == 0)
            {
                Pas[index * (Num + 1) + Num] = a;
                biaoji = 1;
            }

            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Pas[index * (Num + 1) + Num - 1 - j] += temp;
        }
        index++;
    }
}

void work(int g_Num, int g_pasNum, int g_lieNum, int* g_Act, int* g_Pas,
    sycl::nd_item<3> item_ct1)
{
    int g_index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    int gridStride = item_ct1.get_group_range(2) * item_ct1.get_local_range(2);

    for (int i = g_lieNum - 1; i - 8 >= -1; i -= 8)
    {
        for (int j = g_index; j < g_pasNum; j += gridStride)
        {
            while (g_Pas[j * (g_Num + 1) + g_Num] <= i && g_Pas[j * (Num + 1) + g_Num] >= i - 7)
            {
                int index = g_Pas[j * (Num + 1) + g_Num];

                if (g_Act[index * (Num + 1) + g_Num] == 1)
                {
                    for (int k = 0; k < g_Num; k++)
                    {
                        g_Pas[j * (Num + 1) + k] = g_Pas[j * (Num + 1) + k] ^ g_Act[index * (Num + 1) + k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < g_Num; num++)
                    {
                        if (g_Pas[j * (Num + 1) + num] != 0)
                        {
                            unsigned int temp = g_Pas[j * (Num + 1) + num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    g_Pas[j * (Num + 1) + g_Num] = S_num - 1;
                }
                else
                {
                    break;
                }
            }
        }
    }

    for (int i = g_lieNum % 8 - 1; i >= 0; i--)
    {

        for (int j = g_index; j < g_pasNum; j += gridStride)
        {
            while (g_Pas[j * (Num + 1) + g_Num] == i)
            {
                if (g_Act[i * (Num + 1) + g_Num] == 1)
                {
                    for (int k = 0; k < g_Num; k++)
                    {
                        g_Pas[j * (Num + 1) + k] = g_Pas[j * (Num + 1) + k] ^ g_Act[i * (Num + 1) + k];
                    }
                    int num = 0, S_num = 0;
                    for (num = 0; num < g_Num; num++)
                    {
                        if (g_Pas[j * (Num + 1) + num] != 0)
                        {
                            unsigned int temp = g_Pas[j * (Num + 1) + num];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                S_num++;
                            }
                            S_num += num * 32;
                            break;
                        }
                    }
                    g_Pas[j * (Num + 1) + g_Num] = S_num - 1;

                }
                else
                {
                    break;
                }
            }
        }
    }

}


int main() try {
    dpct::device_ext& dev_ct1 = dpct::get_current_device();
    sycl::queue& q_ct1 = dev_ct1.default_queue();
    int ret;

    init_A();
    init_P();


    int* g_Act, * g_Pas;

    ret = (g_Act = sycl::malloc_device<int>(lieNum * (Num + 1), q_ct1), 0);

    ret = (g_Pas = sycl::malloc_device<int>(lieNum * (Num + 1), q_ct1), 0);
    size_t threads_per_block = 256;
    size_t number_of_blocks = 32;

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::steady_clock> start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
    float etime = 0.0;

    start_ct1 = std::chrono::steady_clock::now();
    start = q_ct1.ext_oneapi_submit_barrier();

    bool sign;
    do
    {
        ret =
            (q_ct1.memcpy(g_Act, Act, sizeof(int) * lieNum * (Num + 1)).wait(),
                0);
        ret =
            (q_ct1.memcpy(g_Pas, Pas, sizeof(int) * lieNum * (Num + 1)).wait(),
                0);
        q_ct1.submit([&](sycl::handler& cgh) {
            auto Num_ct0 = Num;
            auto pasNum_ct1 = pasNum;
            auto lieNum_ct2 = lieNum;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 256) *
                sycl::range<3>(1, 1, 32),
                sycl::range<3>(1, 1, 32)),
                [=](sycl::nd_item<3> item_ct1) {
                    work(Num_ct0, pasNum_ct1, lieNum_ct2, g_Act,
                        g_Pas, item_ct1);
                });
            });
        dev_ct1.queues_wait_and_throw();
        ret =
            (q_ct1.memcpy(Act, g_Act, sizeof(int) * lieNum * (Num + 1)).wait(),
                0);
        ret =
            (q_ct1.memcpy(Pas, g_Pas, sizeof(int) * lieNum * (Num + 1)).wait(),
                0);
        sign = false;
        for (int i = 0; i < pasNum; i++)
        {
            int temp = Pas[i * (Num + 1) + Num];
            if (temp == -1)
            {
                continue;
            }
            if (Act[temp * (Num + 1) + Num] == 0)
            {
                for (int k = 0; k < Num; k++)
                    Pas[i * (Num + 1) + Num] = -1;
                sign = true;
            }
        }
    } while (sign == true);

    dpct::get_current_device().queues_wait_and_throw();
    stop_ct1 = std::chrono::steady_clock::now();
    stop = q_ct1.ext_oneapi_submit_barrier();
    etime =
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    printf("GPU_LU:%f ms\n", etime);
}
catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
        << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}