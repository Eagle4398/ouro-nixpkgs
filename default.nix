# default.nix
{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:
pkgs.callPackage ./server.nix { model = "2.6B"; quantization_bits = 4; }
