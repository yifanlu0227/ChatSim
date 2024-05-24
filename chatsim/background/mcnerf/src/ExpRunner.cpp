//
// Created by ppwang on 2022/5/6.
//

#include "ExpRunner.h"
#include <experimental/filesystem>  // GCC 7.5?
#include <fmt/core.h>
#include "Utils/Utils.h"
#include "Utils/cnpy.h"
#include "Utils/StopWatch.h"
#include "Utils/CustomOps/CustomOps.h"
#include <torch/torch.h>

namespace fs = std::experimental::filesystem::v1;
using Tensor = torch::Tensor;


ExpRunner::ExpRunner(const std::string& conf_path) {
  global_data_pool_ = std::make_unique<GlobalDataPool>(conf_path);
  const auto& config = global_data_pool_->config_;
  case_name_ = config["case_name"].as<std::string>();
  base_dir_ = config["base_dir"].as<std::string>();

  base_exp_dir_ = config["base_exp_dir"].as<std::string>();
  global_data_pool_->base_exp_dir_ = base_exp_dir_;
  
  fs::create_directories(base_exp_dir_);

  pts_batch_size_ = config["train"]["pts_batch_size"].as<int>();
  end_iter_ = config["train"]["end_iter"].as<int>();
  vis_freq_ = config["train"]["vis_freq"].as<int>();
  report_freq_ = config["train"]["report_freq"].as<int>();
  stats_freq_ = config["train"]["stats_freq"].as<int>();
  save_freq_ = config["train"]["save_freq"].as<int>();
  learning_rate_ = config["train"]["learning_rate"].as<float>();
  learning_rate_alpha_ = config["train"]["learning_rate_alpha"].as<float>();
  learning_rate_warm_up_end_iter_ = config["train"]["learning_rate_warm_up_end_iter"].as<int>();
  ray_march_init_fineness_ = config["train"]["ray_march_init_fineness"].as<float>();
  ray_march_fineness_decay_end_iter_ = config["train"]["ray_march_fineness_decay_end_iter"].as<int>();
  tv_loss_weight_ = config["train"]["tv_loss_weight"].as<float>();
  disp_loss_weight_ = config["train"]["disp_loss_weight"].as<float>();
  var_loss_weight_ = config["train"]["var_loss_weight"].as<float>();
  var_loss_start_ = config["train"]["var_loss_start"].as<int>();
  var_loss_end_ = config["train"]["var_loss_end"].as<int>();
  gradient_scaling_start_ = config["train"]["gradient_scaling_start"].as<int>();
  gradient_scaling_end_ = config["train"]["gradient_scaling_end"].as<int>();

  // Dataset
  dataset_ = std::make_unique<Dataset>(global_data_pool_.get());

  // Renderer
  renderer_ = std::make_unique<Renderer>(global_data_pool_.get(), dataset_->n_images_);

  // Optimizer
  optimizer_ = std::make_unique<torch::optim::Adam>(renderer_->OptimParamGroups());

  if (config["is_continue"].as<bool>()) {
    LoadCheckpoint(base_exp_dir_ + "/checkpoints/latest");
  }

  if (config["reset"] && config["reset"].as<bool>()) {
    renderer_->Reset();
  }
}



void ExpRunner::Train(bool output_hdr, bool use_shutter) {
  global_data_pool_->mode_ = RunningMode::TRAIN;

  std::string log_dir = base_exp_dir_ + "/logs";
  fs::create_directories(log_dir);

  std::vector<float> mse_records;
  float time_per_iter = 0.f;
  StopWatch clock;

  float psnr_smooth = -1.0;
  UpdateAdaParams();

  {
    StopWatch watch;
    global_data_pool_->iter_step_ = iter_step_;
    for (; iter_step_ < end_iter_;) {
      global_data_pool_->backward_nan_ = false;
      // global_data_pool_->drop_out_prob_ = 1.f - std::min(1.f, float(iter_step_) / 1000.f);
      // global_data_pool_->drop_out_prob_ = 0.f;

      int cur_batch_size = int(pts_batch_size_ / global_data_pool_->meaningful_sampled_pts_per_ray_) >> 4 << 4;
      auto [train_rays, gt_colors_shutter, emb_idx] = dataset_->RandRaysData(cur_batch_size, DATA_TRAIN_SET);
      auto gt_colors = gt_colors_shutter.index({Slc(), Slc(0, 3)});
      auto shutter = gt_colors_shutter.index({Slc(), Slc(3, 4)});
      // std::cout << shutter.sizes() << std::endl;  // [N, 1]

      Tensor& rays_o = train_rays.origins;
      Tensor& rays_d = train_rays.dirs;
      Tensor& bounds = train_rays.bounds;

      auto render_result = renderer_->Render(rays_o, rays_d, bounds, emb_idx);
      Tensor pred_colors = render_result.colors.index({Slc(0, cur_batch_size)});
      Tensor disparity = render_result.disparity;
      
      if (output_hdr) {
        
        ///////////////////////////// exp1107 consider shutter time/////////////////////////////////////
        if (use_shutter) {
          pred_colors = pred_colors * shutter;
        }
        ///////////////////////////////////// gamma correction /////////////////////////////////////////
        pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
        torch::Tensor mask = pred_colors <= 0.0031308;
      
        torch::Tensor low_value = pred_colors * 12.92;
        torch::Tensor high_value = 1.055 * torch::pow(pred_colors, 1 / 2.4) - 0.055;
        
        pred_colors = torch::where(mask, low_value, high_value);

        pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////
      Tensor color_loss = torch::sqrt((pred_colors - gt_colors).square() + 1e-4f).mean();
      Tensor disparity_loss = disparity.square().mean();

      Tensor edge_feats = render_result.edge_feats;
      Tensor tv_loss = (edge_feats.index({Slc(), 0}) - edge_feats.index({Slc(), 1})).square().mean();

      Tensor sampled_weights = render_result.weights;
      Tensor idx_start_end = render_result.idx_start_end;
      Tensor sampled_var = CustomOps::WeightVar(sampled_weights, idx_start_end);
      Tensor var_loss = (sampled_var + 1e-2).sqrt().mean();

      float var_loss_weight = 0.f;
      if (iter_step_ > var_loss_end_) {
        var_loss_weight = var_loss_weight_;
      }
      else if (iter_step_ > var_loss_start_) {
        var_loss_weight = float(iter_step_ - var_loss_start_) / float(var_loss_end_ - var_loss_start_) * var_loss_weight_;
      }

      Tensor loss = color_loss + var_loss * var_loss_weight +
                    disparity_loss * disp_loss_weight_ +
                    tv_loss * tv_loss_weight_;

      float mse = (pred_colors - gt_colors).square().mean().item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      psnr_smooth = psnr_smooth < 0.f ? psnr : psnr * .1f + psnr_smooth * .9f;
      CHECK(!std::isnan(pred_colors.mean().item<float>()));
      CHECK(!std::isnan(gt_colors.mean().item<float>()));
      CHECK(!std::isnan(mse));

      // There can be some cases that the output colors have no grad due to the occupancy grid.
      if (loss.requires_grad()) {
        optimizer_->zero_grad();
        loss.backward();
        if (global_data_pool_->backward_nan_) {
          std::cout << "Nan!" << std::endl;
          continue;
        }
        else {
          optimizer_->step();
        }
      }

      mse_records.push_back(mse);

      iter_step_++;
      global_data_pool_->iter_step_ = iter_step_;

      if (iter_step_ % stats_freq_ == 0) {
        cnpy::npy_save(base_exp_dir_ + "/stats.npy", mse_records.data(), {mse_records.size()});
      }

      if (iter_step_ % vis_freq_ == 0) {
        int t = iter_step_ / vis_freq_;
        int vis_idx;
        vis_idx = (iter_step_ / vis_freq_) % dataset_->test_set_.size();
        vis_idx = dataset_->test_set_[vis_idx];
        VisualizeImage(vis_idx, output_hdr, use_shutter);
      }

      if (iter_step_ % save_freq_ == 0) {
        SaveCheckpoint();
      }
      time_per_iter = time_per_iter * 0.6f + clock.TimeDuration() * 0.4f;

      if (iter_step_ % report_freq_ == 0) {
        std::cout << fmt::format(
            "Iter: {:>6d} PSNR: {:.2f} NRays: {:>5d} OctSamples: {:.1f} Samples: {:.1f} MeaningfulSamples: {:.1f} IPS: {:.1f} LR: {:.4f}",
            iter_step_,
            psnr_smooth,
            cur_batch_size,
            global_data_pool_->sampled_oct_per_ray_,
            global_data_pool_->sampled_pts_per_ray_,
            global_data_pool_->meaningful_sampled_pts_per_ray_,
            1.f / time_per_iter,
            optimizer_->param_groups()[0].options().get_lr())
                  << std::endl;
      }
      UpdateAdaParams();
    }
    YAML::Node info_data;

    std::ofstream info_fout(base_exp_dir_ + "/train_info.txt");
    info_fout << watch.TimeDuration() << std::endl;
    info_fout.close();
  }

  std::cout << "Train done, test." << std::endl;
  TestImages(output_hdr, use_shutter);
}

void ExpRunner::LoadCheckpoint(const std::string& path) {
  {
    Tensor scalars;
    std::cout << (scalars, path + "/scalars.pt") << std::endl;
    torch::load(scalars, path + "/scalars.pt");
    iter_step_ = std::round(scalars[0].item<float>());
    UpdateAdaParams();
  }

  {
    std::vector<Tensor> scene_states;
    torch::load(scene_states, path + "/renderer.pt");
    renderer_->LoadStates(scene_states, 0);
  }
}

void ExpRunner::SaveCheckpoint() {
  std::string output_dir = base_exp_dir_ + fmt::format("/checkpoints/{:0>8d}", iter_step_);
  fs::create_directories(output_dir);

  fs::remove_all(base_exp_dir_ + "/checkpoints/latest");
  fs::create_directory(base_exp_dir_ + "/checkpoints/latest");
  // scene
  torch::save(renderer_->States(), output_dir + "/renderer.pt");
  fs::create_symlink(output_dir + "/renderer.pt", base_exp_dir_ + "/checkpoints/latest/renderer.pt");
  // optimizer
  // torch::save(*(optimizer_), output_dir + "/optimizer.pt");
  // other scalars
  Tensor scalars = torch::empty({1}, CPUFloat);
  scalars.index_put_({0}, float(iter_step_));
  torch::save(scalars, output_dir + "/scalars.pt");
  fs::create_symlink(output_dir + "/scalars.pt", base_exp_dir_ + "/checkpoints/latest/scalars.pt");
}

void ExpRunner::UpdateAdaParams() {
  // Update ray march fineness
  if (iter_step_ >= ray_march_fineness_decay_end_iter_) {
    global_data_pool_->ray_march_fineness_ = 1.f;
  }
  else {
    float progress = float(iter_step_) / float(ray_march_fineness_decay_end_iter_);
    global_data_pool_->ray_march_fineness_ = std::exp(std::log(1.f) * progress + std::log(ray_march_init_fineness_) * (1.f - progress));
  }
  // Update learning rate
  float lr_factor;
  if (iter_step_ >= learning_rate_warm_up_end_iter_) {
    float progress = float(iter_step_ - learning_rate_warm_up_end_iter_) /
                     float(end_iter_ - learning_rate_warm_up_end_iter_);
    lr_factor = (1.f - learning_rate_alpha_) * (std::cos(progress * float(M_PI)) * .5f + .5f) + learning_rate_alpha_;
  }
  else {
    lr_factor = float(iter_step_) / float(learning_rate_warm_up_end_iter_);
  }
  float lr = learning_rate_ * lr_factor;
  for (auto& g : optimizer_->param_groups()) {
    g.options().set_lr(lr);
  }

  // Update gradient scaling ratio
  {
    float progress = 1.f;
    if (iter_step_ < gradient_scaling_end_) {
      progress = std::max(0.f,
          (float(iter_step_) - gradient_scaling_start_) / (gradient_scaling_end_ - gradient_scaling_start_ + 1e-9f));
    }
    global_data_pool_->gradient_scaling_progress_ = progress;
  }
}


std::tuple<Tensor, Tensor, Tensor> ExpRunner::RenderWholeImage(Tensor rays_o, Tensor rays_d, Tensor bounds) {
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor first_oct_disp = torch::full({n_rays, 1}, 1.f, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds = bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result = renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor());
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    // colors = torch::clamp(colors, 0.0, 1.0);
      
    // torch::Tensor mask = colors <= 0.0031308;
  
    // torch::Tensor low_value = colors * 12.92;
    // torch::Tensor high_value = 1.055 * torch::pow(colors, 1 / 2.4) - 0.055;
    
    // colors = torch::where(mask, low_value, high_value);

    // colors = torch::clamp(colors, 0.0, 1.0);
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    pred_colors.index_put_({Slc(i, i_high)}, colors);
    
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor& ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis = render_result.first_oct_dis.detach().to(torch::kCPU);
        first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
      }
    }
  }
  pred_disp = pred_disp / pred_disp.max();
  first_oct_disp = first_oct_disp.min() / first_oct_disp;

  return { pred_colors, first_oct_disp, pred_disp };
}

std::tuple<Tensor, Tensor, Tensor> ExpRunner::RenderWholeImage_weights(Tensor rays_o, Tensor rays_d, Tensor bounds) {
  torch::NoGradGuard no_grad_guard;
  rays_o = rays_o.to(torch::kCPU);
  rays_d = rays_d.to(torch::kCPU);
  bounds = bounds.to(torch::kCPU);
  const int n_rays = rays_d.sizes()[0];

  Tensor pred_colors = torch::zeros({n_rays, 3}, CPUFloat);
  Tensor first_oct_disp = torch::full({n_rays, 1}, 1.f, CPUFloat);
  Tensor pred_disp = torch::zeros({n_rays, 1}, CPUFloat);
  Tensor pred_last_trans = torch::zeros({n_rays, 1}, CPUFloat);

  const int ray_batch_size = 8192;
  for (int i = 0; i < n_rays; i += ray_batch_size) {
    int i_high = std::min(i + ray_batch_size, n_rays);
    Tensor cur_rays_o = rays_o.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_rays_d = rays_d.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();
    Tensor cur_bounds = bounds.index({Slc(i, i_high)}).to(torch::kCUDA).contiguous();

    auto render_result = renderer_->Render(cur_rays_o, cur_rays_d, cur_bounds, Tensor());
    Tensor colors = render_result.colors.detach().to(torch::kCPU);
    // std::cout << render_result.colors.sizes() << std::endl;
    // std::cout << render_result.weights.sizes() << std::endl;
    Tensor disp = render_result.disparity.detach().to(torch::kCPU).squeeze();
    Tensor last_trans_ = render_result.last_trans.detach().to(torch::kCPU).squeeze();
    // std::cout << "last_trans" << last_trans.sizes() << std::endl;
    // Tensor weights_ = render_result.weights.detach().to(torch::kCPU).squeeze();
    // std::cout << weights_.sizes() << std::endl;
    // pred_weights.index_put_({Slc(i, i_high)}, weights_.unsqueeze(-1));
    

    pred_colors.index_put_({Slc(i, i_high)}, colors);
    pred_disp.index_put_({Slc(i, i_high)}, disp.unsqueeze(-1));
    pred_last_trans.index_put_({Slc(i, i_high)}, last_trans_.unsqueeze(-1));
    
    if (!render_result.first_oct_dis.sizes().empty()) {
      Tensor& ret_first_oct_dis = render_result.first_oct_dis;
      if (ret_first_oct_dis.has_storage()) {
        Tensor cur_first_oct_dis = render_result.first_oct_dis.detach().to(torch::kCPU);
        first_oct_disp.index_put_({Slc(i, i_high)}, cur_first_oct_dis);
      }
    }

  }
  // std::cout << pred_weights.sizes() << std::endl;
  pred_disp = pred_disp / pred_disp.max();
  first_oct_disp = first_oct_disp.min() / first_oct_disp;
  Tensor save_pred_last_trans = pred_last_trans.reshape({1280, 1280*4, 1});
  torch::save(save_pred_last_trans, base_exp_dir_ + "/last_trans.pt");

  return { pred_colors, first_oct_disp, pred_disp };
}

void ExpRunner::RenderAllImages() {
  for (int idx = 0; idx < dataset_->n_images_; idx++) {
    VisualizeImage(idx, false, false);
  }
}

void ExpRunner::VisualizeImage(int idx, bool output_hdr, bool use_shutter) {
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  auto [ rays_o, rays_d, bounds ] = dataset_->RaysOfCamera(idx);
  auto [ pred_colors, first_oct_dis, pred_disps ] = RenderWholeImage(rays_o, rays_d, bounds);

  int H = dataset_->height_;
  int W = dataset_->width_;

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  if (output_hdr){
    auto shutter = dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 4}).index({Slc(), Slc(), Slc(3, 4)});
    
    if (use_shutter){
      pred_colors = pred_colors.reshape({H, W, 3}) * shutter;
      std::cout << "use shutter" << std::endl;
    }
    pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
    torch::Tensor mask = pred_colors <= 0.0031308;
  
    torch::Tensor low_value = pred_colors * 12.92;
    torch::Tensor high_value = 1.055 * torch::pow(pred_colors, 1 / 2.4) - 0.055;
    
    pred_colors = torch::where(mask, low_value, high_value);

    pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////////

  Tensor img_tensor = torch::cat({dataset_->image_tensors_[idx].to(torch::kCPU).reshape({H, W, 4}).index({Slc(), Slc(), Slc(0, 3)}),
                                  pred_colors.reshape({H, W, 3}),
                                  first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
                                  pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);
  fs::create_directories(base_exp_dir_ + "/images");
  Utils::WriteImageTensor(base_exp_dir_ + "/images/" + fmt::format("{}_{}.png", iter_step_, idx), img_tensor);

  global_data_pool_->mode_ = prev_mode;
}

// void ExpRunner::RenderPath() {
//   std::cout << "start rendering path" << std::endl;
//   torch::NoGradGuard no_grad_guard;
//   int n_images = dataset_->render_poses_.size(0);
//   global_data_pool_->mode_ = RunningMode::VALIDATE;
//   int res_level = 1;
//   for (int i = 0; i < n_images; i++) {
//     std::cout << i << std::endl;
//     auto [ rays_o, rays_d, bounds ] = dataset_->RaysFromPose(dataset_->render_poses_[i], res_level);
//     // torch::save(rays_o, "/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/panorama/tmp/rays_o_tmp.pt");
//     // torch::save(rays_d, "/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/panorama/tmp/rays_d_tmp.pt");
//     torch::save(bounds, base_exp_dir_ + "/bounds_tmp.pt");
//     auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds);
//     int H = dataset_->height_ / res_level;
//     int W = dataset_->width_ / res_level;

//     Tensor img_tensor = torch::cat({pred_colors.reshape({H, W, 3}),
//                                     first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
//                                     pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);

//     fs::create_directories(base_exp_dir_ + "/novel_images");
//     Utils::WriteImageTensor(base_exp_dir_ + "/novel_images/" + fmt::format("{}_{:0>3d}.png", iter_step_, i), img_tensor);
//   }
// }

void ExpRunner::RenderPath(bool output_hdr, bool use_shutter, bool wide_angle) {
  std::cout << "start rendering wide angle images" << std::endl;
  torch::NoGradGuard no_grad_guard;
  int n_images = dataset_->render_poses_.size(0);
  global_data_pool_->mode_ = RunningMode::VALIDATE;
  int res_level = 1;
  for (int i = 0; i < n_images; i++) {
    std::cout << i << std::endl;

    auto [ rays_o, rays_d, bounds ] = dataset_->RaysFromPose_wide_angle(dataset_->render_poses_[i], res_level);
    if (! wide_angle) {
      auto [ _rays_o, _rays_d, _bounds ] = dataset_->RaysFromPose(dataset_->render_poses_[i], res_level);
      rays_o = _rays_o;
      rays_d = _rays_d;
      bounds = _bounds;
    }

    auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds);
    int H = dataset_->height_ / res_level;
    int W = dataset_->width_ / res_level;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    if (output_hdr){
      
      if (use_shutter){
        // use the first frame's shutter
        pred_colors = pred_colors * (dataset_->image_tensors_[0].to(torch::kCPU).reshape({H, W, 4}).index({Slc(0,1), Slc(0,1), Slc(3,4)}));
        std::cout << "use shutter" << std::endl;
      }
      pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
      torch::Tensor mask = pred_colors <= 0.0031308;
    
      torch::Tensor low_value = pred_colors * 12.92;
      torch::Tensor high_value = 1.055 * torch::pow(pred_colors, 1 / 2.4) - 0.055;
      
      pred_colors = torch::where(mask, low_value, high_value);

      pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    if (wide_angle) {
      W *= 3;
    }
    Tensor img_tensor = torch::cat({pred_colors.reshape({H, W, 3}),
                                    first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
                                    pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);
    if (wide_angle) {
      fs::create_directories(base_exp_dir_ + "/wide_angle_novel_images");
      Utils::WriteImageTensor(base_exp_dir_ + "/wide_angle_novel_images/" + fmt::format("{}_{:0>3d}.png", iter_step_, i), img_tensor);
    } else {
      fs::create_directories(base_exp_dir_ + "/novel_images");
      Utils::WriteImageTensor(base_exp_dir_ + "/novel_images/" + fmt::format("{}_{:0>3d}.png", iter_step_, i), img_tensor.index({Slc(), Slc(0, W), Slc()}));
    }
  }
}

// void ExpRunner::RenderPanorama() {
//   torch::NoGradGuard no_grad_guard;
//   int n_images = dataset_->render_poses_.size(0);
//   global_data_pool_->mode_ = RunningMode::VALIDATE;
//   int res_level = 1;
//   for (int i = 0; i < n_images; i++) {
//     std::cout << i << std::endl;
//     auto [ rays_o, rays_d, bounds ] = dataset_->RaysFromPose(dataset_->render_poses_[i], res_level);
//     // torch::save(rays_o, "/home/yfl/workspace/f2-nerf/panorama/tmp/rays_o_tmp.pt");
//     // torch::save(rays_d, "/home/yfl/workspace/f2-nerf/panorama/tmp/rays_d_tmp.pt");
//     // torch::save(bounds, "/home/yfl/workspace/f2-nerf/panorama/tmp/bounds_tmp.pt");
//     cnpy::NpyArray rays_d_ = cnpy::npy_load("/dssg/home/acct-umjpyb/umjpyb/ziwang/f2-nerf/panorama/tmp/rays_d_frontview.npy");
//     auto options = torch::TensorOptions().dtype(torch::kFloat32);  // WARN: Float64 Here!!!!!
//     std::vector<int64_t> shape_rays_d(rays_d_.shape.begin(), rays_d_.shape.end());
//     Tensor rays_d_frontview = torch::from_blob(rays_d_.data<float>(), torch::IntArrayRef(shape_rays_d), options).to(torch::kFloat32).to(torch::kCUDA);
//     // rays_d_frontview.reshape({dataset_->height_*dataset_->width_, 3});
//     std::cout << rays_d_frontview.index({0, 0}) << std::endl;
//     std::cout << rays_d_frontview.index({0, 1}) << std::endl;
//     std::cout << rays_d_frontview.index({0, 2}) << std::endl;
//     auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d_frontview, bounds);
//     int H = dataset_->height_ / res_level;
//     int W = dataset_->width_ / res_level;

//     Tensor img_tensor = torch::cat({pred_colors.reshape({H, W, 3}),
//                                     first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
//                                     pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);

//     fs::create_directories(base_exp_dir_ + "/panorama");
//     Utils::WriteImageTensor(base_exp_dir_ + "/panorama/" + fmt::format("{}_{:0>3d}.png", iter_step_, i), img_tensor);
//   }
// }

void ExpRunner::RenderPanorama(bool use_shutter) {
  torch::NoGradGuard no_grad_guard;
  // int n_images = dataset_->render_poses_.size(0);
  int n_images = 1;
  global_data_pool_->mode_ = RunningMode::VALIDATE;
  int res_level = 1;
  for (int i = 0; i < n_images; i++) {
    std::cout << i << std::endl;
    // auto [ rays_o, rays_d, bounds ] = dataset_->RaysFromPose(dataset_->render_poses_[i], res_level);
    // Tensor rays_o;
    // torch::load(rays_o, "/home/yfl/workspace/f2-nerf/panorama/tmp/rays_o.pt");
    // std::cout << rays_o << std::endl;
    // Tensor rays_d;
    // torch::load(rays_d, "/home/yfl/workspace/f2-nerf/panorama/tmp/rays_d.pt");
    // Tensor bounds;
    // torch::load(bounds, "/home/yfl/workspace/f2-nerf/panorama/tmp/bounds.pt");
    cnpy::NpyArray rays_o_ = cnpy::npy_load(base_exp_dir_ + "/rays_o.npy");
    cnpy::NpyArray rays_d_ = cnpy::npy_load(base_exp_dir_ + "/rays_d.npy");
    cnpy::NpyArray bounds_ = cnpy::npy_load(base_exp_dir_ + "/bounds.npy");
    // cnpy::NpyArray rays_o_ = cnpy::npy_load("/home/yfl/workspace/f2-nerf/panorama/tmp/rays_o_ori.npy");
    // cnpy::NpyArray rays_d_ = cnpy::npy_load("/home/yfl/workspace/f2-nerf/panorama/tmp/rays_d_ori.npy");
    // cnpy::NpyArray bounds_ = cnpy::npy_load("/home/yfl/workspace/f2-nerf/panorama/tmp/bounds_ori.npy");
    
    // torch::IntArrayRef rays_o_shape(rays_o_.shape);
    // torch::IntArrayRef rays_d_shape(rays_d_.shape);
    // torch::IntArrayRef bounds_shape(bounds_.shape);
    auto options = torch::TensorOptions().dtype(torch::kFloat32);  
    std::vector<int64_t> shape_rays_o(rays_o_.shape.begin(), rays_o_.shape.end());
    Tensor rays_o_from_numpy = torch::from_blob(rays_o_.data<float>(), torch::IntArrayRef(shape_rays_o), options).to(torch::kFloat32).to(torch::kCUDA);
    std::vector<int64_t> shape_rays_d(rays_d_.shape.begin(), rays_d_.shape.end());
    Tensor rays_d_from_numpy = torch::from_blob(rays_d_.data<float>(), torch::IntArrayRef(shape_rays_d), options).to(torch::kFloat32).to(torch::kCUDA);
    std::vector<int64_t> shape_bounds(bounds_.shape.begin(), bounds_.shape.end());
    Tensor bounds = torch::from_blob(bounds_.data<float>(), torch::IntArrayRef(shape_bounds), options).to(torch::kFloat32).to(torch::kCUDA);
    // std::cout << rays_o.sizes() << std::endl;
    // std::cout << rays_d.sizes() << std::endl;
    // std::cout << bounds.sizes() << std::endl;
    // std::cout << rays_d.index({0, 0}) << std::endl;
    // std::cout << rays_d.index({0, 1}) << std::endl;
    // std::cout << rays_d.index({0, 2}) << std::endl;

    // float near = bounds.index({Slc(), 0}).item<float>();
    // float far  = bounds.index({Slc(), 1}).item<float>();
    // Tensor bounds_ = torch::stack({
    //   torch::full({ 1280 * 1280 * 4 }, near, CUDAFloat),
    //   torch::full({ 1280 * 1280 * 4 }, far,  CUDAFloat)
    // }, -1).contiguous();
    
    auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage_weights(rays_o_from_numpy, rays_d_from_numpy, bounds);
    int H = dataset_->height_ / res_level;
    int W = dataset_->width_ / res_level;
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    if (use_shutter){
      pred_colors = pred_colors * (dataset_->image_tensors_[0].to(torch::kCPU).reshape({H, W, 4}).index({Slc(0,1), Slc(0,1), Slc(3,4)}));
      std::cout << "use shutter" << std::endl;
    }
    // pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
    // torch::Tensor mask = pred_colors <= 0.0031308;
      
    // torch::Tensor low_value = pred_colors * 12.92;
    // torch::Tensor high_value = 1.055 * torch::pow(pred_colors, 1 / 2.4) - 0.055;
    
    // pred_colors = torch::where(mask, low_value, high_value);

    // pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    

    // Tensor img_tensor = torch::cat({pred_colors.reshape({H, W, 3}),
    //                                 first_oct_dis.reshape({H, W, 1}).repeat({1, 1, 3}),
    //                                 pred_disps.reshape({H, W, 1}).repeat({1, 1, 3})}, 1);
    Tensor img_tensor = pred_colors.reshape({1280, 1280 * 4, 3});
    torch::save(img_tensor, base_exp_dir_ + "/nerf_panorama.pt");
    fs::create_directories(base_exp_dir_ + "/panorama");
    Utils::WriteImageTensor(base_exp_dir_ + "/panorama/" + fmt::format("{}_{:0>3d}.png", iter_step_, i), img_tensor);
  }
}

void ExpRunner::TestImages(bool output_hdr, bool use_shutter) {
  torch::NoGradGuard no_grad_guard;
  auto prev_mode = global_data_pool_->mode_;
  global_data_pool_->mode_ = RunningMode::VALIDATE;

  float psnr_sum = 0.f;
  float cnt = 0.f;
  YAML::Node out_info;
  {
    fs::create_directories(base_exp_dir_ + "/test_images");
    for (int i: dataset_->test_set_) {
      auto [rays_o, rays_d, bounds] = dataset_->RaysOfCamera(i);
      auto [pred_colors, first_oct_dis, pred_disps] = RenderWholeImage(rays_o, rays_d, bounds);  // At this stage, the returned number is

      int H = dataset_->height_;
      int W = dataset_->width_;

      auto quantify = [](const Tensor& x) {
        return (x.clip(0.f, 1.f) * 255.f).to(torch::kUInt8).to(torch::kFloat32) / 255.f;
      };
      pred_disps = pred_disps.reshape({H, W, 1});
      first_oct_dis = first_oct_dis.reshape({H, W, 1});
      pred_colors = pred_colors.reshape({H, W, 3});
      ///////////////////////////////////////////////////////////////////////////////////////////////////
      if (output_hdr){
        auto shutter = dataset_->image_tensors_[i].to(torch::kCPU).reshape({H, W, 4}).index({Slc(), Slc(), Slc(3, 4)});
        
        if (use_shutter){
          pred_colors = pred_colors.reshape({H, W, 3}) * shutter;
          std::cout << "use shutter" << std::endl;
        }
        pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
        torch::Tensor mask = pred_colors <= 0.0031308;
      
        torch::Tensor low_value = pred_colors * 12.92;
        torch::Tensor high_value = 1.055 * torch::pow(pred_colors, 1 / 2.4) - 0.055;
        
        pred_colors = torch::where(mask, low_value, high_value);

        pred_colors = torch::clamp(pred_colors, 0.0, 1.0);
      }
      ///////////////////////////////////////////////////////////////////////////////////////////////////
      pred_colors = quantify(pred_colors);
      // std::cout << dataset_->image_tensors_[i].sizes() << std::endl;    // [1280, 1920, 4]
      // float mse = (pred_colors.reshape({H, W, 3}) -
      //              dataset_->image_tensors_[i].to(torch::kCPU).reshape({H, W, 3})).square().mean().item<float>();
      // exp1107
      float mse = (pred_colors.reshape({H, W, 3}) -
                   dataset_->image_tensors_[i].to(torch::kCPU).reshape({H, W, 4}).index({Slc(), Slc(), Slc(0, 3)})).square().mean().item<float>();
      float psnr = 20.f * std::log10(1 / std::sqrt(mse));
      out_info[fmt::format("{}", i)] = psnr;
      std::cout << fmt::format("{}: {}", i, psnr) << std::endl;
      psnr_sum += psnr;
      cnt += 1.f;
      Utils::WriteImageTensor(base_exp_dir_ + "/test_images/" + fmt::format("color_{}_{:0>3d}.png", iter_step_, i),
                             pred_colors);
      Utils::WriteImageTensor(base_exp_dir_ + "/test_images/" + fmt::format("depth_{}_{:0>3d}.png", iter_step_, i),
                              pred_disps.repeat({1, 1, 3}));
      Utils::WriteImageTensor(base_exp_dir_ + "/test_images/" + fmt::format("oct_depth_{}_{:0>3d}.png", iter_step_, i),
                             first_oct_dis.repeat({1, 1, 3}));

    }
  }
  float mean_psnr = psnr_sum / cnt;
  std::cout << fmt::format("Mean psnr: {}", mean_psnr) << std::endl;
  out_info["mean_psnr"] = mean_psnr;

  std::ofstream info_fout(base_exp_dir_ + "/test_images/info.yaml");
  info_fout << out_info;

  global_data_pool_->mode_ = prev_mode;
}

void ExpRunner::Execute() {
  std::string mode = global_data_pool_->config_["mode"].as<std::string>();
  if (mode == "train") {
    Train(false, false);
  }
  else if (mode == "train_hdr") {
    Train(true, false);
  }
  else if (mode == "train_hdr_shutter") {
    Train(true, true);
  }
  else if (mode == "render_path") {
    RenderPath(false, false, false);
  }
  else if (mode == "render_path_hdr_shutter") {
    RenderPath(true, true, false);
  }
  else if (mode == "test") {
    TestImages(false, false);
  }
  else if (mode == "test_hdr") {
    TestImages(true, false);
  }
  else if (mode == "test_hdr_shutter") {
    TestImages(true, true);
  }
  else if (mode == "render_all") {
    RenderAllImages();
  }
  else if (mode == "render_panorama") {
    RenderPanorama(false);
  }
  else if (mode == "render_panorama_shutter") {
    RenderPanorama(true);
  }
  else if (mode == "render_wide_angle") {
    RenderPath(false, false, true);
  }
  else if (mode == "render_wide_angle_hdr") {
    RenderPath(true, false, true);
  }
  else if (mode == "render_wide_angle_hdr_shutter") {
    RenderPath(true, true, true);
  }
  else if (mode == "render_hdr_shutter"){
    RenderPath(true, true, false);
  }
  else {
    std::cout << "Unknown mode: " << mode << std::endl;
  }
  
}
