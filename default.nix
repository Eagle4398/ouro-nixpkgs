# { pkgs ? import (fetchTarball
#   "https://github.com/NixOS/nixpkgs/archive/cb369ef2efd432b3cdf8622b0ffc0a97a02f3137.tar.gz") {
#     config.allowUnfree = true;
#   } }:
# pkgs.callPackage ./server.nix {
#   model = "2.6B";
#   quantization_bits = 4;
# }
# # default.nix
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
pkgs.callPackage ./server.nix {
  model = "2.6B";
  quantization_bits = 4;
}
