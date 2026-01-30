import torch

#
# A fatal error has been detected by the Java Runtime Environment:
#
#  SIGSEGV (0xb) at pc=0x00007fc8ba5b3cc4, pid=7916, tid=0x00007fc929d07700
#
# JRE version: OpenJDK Runtime Environment (8.0_242-b08) (build 1.8.0_242-8u242-b08-0ubuntu3~16.04-b08)
# Java VM: OpenJDK 64-Bit Server VM (25.242-b08 mixed mode linux-amd64 compressed oops)
# Problematic frame:
# C  [libtorch.so+0x62a7cc4]  torch::autograd::VariableType::(anonymous namespace)::div(at::Tensor const&, at::Tensor const&)+0xaa4
#
# Failed to write core dump. Core dumps have been disabled. To en

# C  [libtorch_cpu.so+0x11b5a4c]  c10::impl::InlineDeviceGuard<c10::impl::VirtualGuardImpl>::InlineDeviceGuard(c10::Device)+0x2c