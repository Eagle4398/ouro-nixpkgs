{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = [ pkgs.cudaPackages.cudatoolkit ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
  '';
}
