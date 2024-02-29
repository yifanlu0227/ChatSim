#include <iostream>
#include <memory>
#include <torch/torch.h>
#include "src/ExpRunner.h"
#include <time.h>

int main(int argc, char* argv[]) {
  std::cout << "Aoligei!" << std::endl;
  torch::manual_seed(2022);

  std::string conf_path = "./runtime_config.yaml";
  auto exp_runner = std::make_unique<ExpRunner>(conf_path);
  // clock_t start,end;
  // start = clock();
  exp_runner->Execute();
  // end = clock();
  // std::cout<<"time = "<<double(end-start)/CLOCKS_PER_SEC<<"s"<< std::endl;
  return 0;
}
