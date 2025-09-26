#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <Eigen/Dense>
#include <thread>
#include <mutex>

#include "optimized_ICP_GN.h"

// 声明UpdateVisualization函数以便在main.cpp中使用
void UpdateVisualization(boost::shared_ptr<pcl::visualization::PCLVisualizer> &viewer,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        const std::string &id,
                        int iteration,
                        const Eigen::Vector3i &color = Eigen::Vector3i(255, 0, 0),
                        int viewport = 0);

using namespace std;

int main()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_opti_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source_svd_transformed_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::io::loadPCDFile("../data/room_scan2.pcd", *cloud_target_ptr);
    pcl::io::loadPCDFile("../data/room_scan1.pcd", *cloud_source_ptr);

    Eigen::Matrix4f T_predict, T_final;
    T_predict.setIdentity();
    T_predict << 0.765, 0.643, -0.027, -1.472,
        -0.6, 0.765, -0.023, 1.3,
        0.006, 0.035, 0.999, -0.1,
        0, 0, 0, 1;

    std::cout << "Wait, matching..." << std::endl;

    // =======================   创建分屏可视化器   =======================
    boost::shared_ptr<pcl::visualization::PCLVisualizer> process_viewer(
        new pcl::visualization::PCLVisualizer("ICP Registration Process - Split View"));
    
    // 创建两个视口：左侧显示优化ICP，右侧显示SVD-ICP
    int v1(0), v2(1);
    process_viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    process_viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    
    // 设置背景色
    process_viewer->setBackgroundColor(0, 0, 0, v1);
    process_viewer->setBackgroundColor(0, 0, 0, v2);
    
    // 在两个视口中添加目标点云（蓝色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_v1(
        cloud_target_ptr, 0, 0, 255);
    process_viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr, target_color_v1, "target_v1", v1);
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_v2(
        cloud_target_ptr, 0, 0, 255);
    process_viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr, target_color_v2, "target_v2", v2);
    
    // 设置点云大小
    process_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_v1", v1);
    process_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_v2", v2);
    
    // 添加坐标系
    process_viewer->addCoordinateSystem(1.0, "coord_v1", v1);
    process_viewer->addCoordinateSystem(1.0, "coord_v2", v2);
    
    // 添加标题
    process_viewer->addText("Optimized ICP (GN)", 10, 30, "title_v1", v1);
    process_viewer->addText("SVD ICP", 10, 30, "title_v2", v2);
    
    // 设置相机位置
    process_viewer->initCameraParameters();
    process_viewer->setCameraPosition(0, 0, 20, 0, 10, 10, v1);
    process_viewer->setCameraPosition(0, 0, 20, 0, 10, 10, v2);
    
    // 使用互斥锁保护可视化器的访问
    std::mutex viewer_mutex;
    
    // 标志变量表示各线程是否完成
    bool optimized_icp_done = false;
    bool svd_icp_done = false;
    
    // 存储配准结果的变量
    Eigen::Matrix4f T_optimal = T_predict;
    Eigen::Matrix4f T_svd = T_predict;
    
    // 优化ICP线程函数
    auto optimized_icp_thread = [&]() {
        std::cout << "Starting Optimized ICP thread..." << std::endl;
        OptimizedICPGN optimized_icp;
        optimized_icp.SetMaxIterations(30);
        optimized_icp.SetMaxCorrespondDistance(0.3);
        optimized_icp.SetTransformationEpsilon(1e-4);
        optimized_icp.SetTargetCloud(cloud_target_ptr);
        
        // 直接调用MatchWithVisualization，函数内部已有适当的互斥锁保护
        optimized_icp.MatchWithVisualization(cloud_source_ptr, T_predict, 
                                            cloud_source_opti_transformed_ptr, 
                                            T_optimal, process_viewer, 1, v1);
        
        optimized_icp_done = true;
        std::cout << "Optimized ICP thread completed." << std::endl;
    };
    
    // SVD-ICP线程函数
    auto svd_icp_thread = [&]() {
        std::cout << "Starting SVD ICP thread..." << std::endl;
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_svd;
        icp_svd.setInputTarget(cloud_target_ptr);
        icp_svd.setInputSource(cloud_source_ptr);
        icp_svd.setMaxCorrespondenceDistance(0.3);
        icp_svd.setMaximumIterations(30);
        icp_svd.setEuclideanFitnessEpsilon(1e-4);
        icp_svd.setTransformationEpsilon(1e-4);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_svd_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        for (int i = 0; i < 30; ++i) {
            // 执行一次迭代
            icp_svd.align(*temp_svd_cloud, T_svd);
            T_svd = icp_svd.getFinalTransformation();
            
            // 更新可视化（每次迭代都显示）
            {   
                std::lock_guard<std::mutex> lock(viewer_mutex);
                UpdateVisualization(process_viewer, temp_svd_cloud, "svd_iteration", i, Eigen::Vector3i(0, 255, 0), v2);
                process_viewer->spinOnce(10);
            }
            
            // 检查收敛条件
            if (icp_svd.hasConverged()) {
                break;
            }
        }
        
        svd_icp_done = true;
        std::cout << "SVD ICP thread completed." << std::endl;
    };
    
    // 创建并启动两个线程
    std::thread thread1(optimized_icp_thread);
    std::thread thread2(svd_icp_thread);
    
    // 等待两个线程完成
    thread1.join();
    thread2.join();
    
    // 线程已经通过join完成，执行必要的可视化更新
    {   
        std::lock_guard<std::mutex> lock(viewer_mutex);
        process_viewer->spinOnce(10);
    }
    
    // 等待两个线程完成
    if (thread1.joinable()) {
        thread1.join();
    }
    if (thread2.joinable()) {
        thread2.join();
    }
    
    // 保存最终结果
    pcl::transformPointCloud(*cloud_source_ptr, *cloud_source_svd_transformed_ptr, T_svd);
    
    // 显示最终结果
    UpdateVisualization(process_viewer, cloud_source_svd_transformed_ptr, "svd_final", 30, Eigen::Vector3i(0, 255, 0), v2);
    
    // 输出配准结果信息
    std::cout << "\n============== Optimized ICP =================" << std::endl;
    std::cout << "T final: \n" << T_optimal << std::endl;
    // 重新创建对象计算fitness score
    OptimizedICPGN optimized_icp_for_score;
    optimized_icp_for_score.SetTargetCloud(cloud_target_ptr);
    std::cout << "fitness score: " << optimized_icp_for_score.GetFitnessScore(cloud_source_ptr) << std::endl;
    
    std::cout << "\n============== SVD ICP =================" << std::endl;
    std::cout << "T final: \n" << T_svd << std::endl;
    // 重新创建对象计算fitness score
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_svd_for_score;
    icp_svd_for_score.setInputTarget(cloud_target_ptr);
    icp_svd_for_score.setInputSource(cloud_source_ptr);
    icp_svd_for_score.align(*cloud_source_svd_transformed_ptr, T_svd);
    std::cout << "fitness score: " << icp_svd_for_score.getFitnessScore() << std::endl;
    
    std::cout << "Press 'q' in the visualization window to continue..." << std::endl;
    
    // 等待用户关闭可视化窗口
    while (!process_viewer->wasStopped()) {
        process_viewer->spinOnce(100);
    }
    // =======================   分屏可视化配准过程   =======================

    // 可视化
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));
    viewer->initCameraParameters();

    // 使用之前定义的v1和v2变量
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->setBackgroundColor(0, 0, 0, v1);
    viewer->addText("Optimized ICP", 10, 10, "optimized icp", v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_opti_color(
        cloud_source_opti_transformed_ptr, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_source_opti_transformed_ptr, source_opti_color, "source opti cloud", v1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_0(cloud_target_ptr, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr, target_color_0, "target cloud1", v1);

    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer->setBackgroundColor(0.0, 0.0, 0.0, v2);
    viewer->addText("SVD ICP", 10, 10, "svd icp", v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_1(cloud_target_ptr, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_target_ptr, target_color_1, "target cloud2", v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_svd_color(cloud_source_svd_transformed_ptr,
                                                                                     0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(cloud_source_svd_transformed_ptr, source_svd_color, "source svd cloud", v2);

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source opti cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source svd cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target cloud2");
    viewer->addCoordinateSystem(1.0);

    viewer->setCameraPosition(0, 0, 20, 0, 10, 10, v1);
    viewer->setCameraPosition(0, 0, 20, 0, 10, 10, v2);

    viewer->spin();

    return 0;
}