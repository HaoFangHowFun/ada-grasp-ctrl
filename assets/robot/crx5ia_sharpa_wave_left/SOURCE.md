# CRX-5iA + left Sharpa Wave preview asset

This directory is a local, display-oriented import of one arm and one hand. It is not registered as an Ada Grasp Ctrl control hand.

## Source snapshot

| Component | Local source | Revision / recorded source |
| --- | --- | --- |
| Assembly | `/mnt/c/Users/Howard/bimanual_manipulation/bimanual_manipulation/composite_description.py` | `a7a10b2e75b88ffa3b2704266d57041a0c3e497b` |
| Arm mounting tree | `/mnt/c/Users/Howard/dual_crx_control/urdf/dual_crx.urdf.xacro` | `c0c25931b62f408eb0ed238d40ef305efd16387c` |
| FANUC CRX-5iA macro and meshes | `/home/howard/ws_fanuc/src/fanuc_description/fanuc_crx_description/`; Git origin: `https://github.com/FANUC-CORPORATION/fanuc_description.git` | `fb40c9803a826ba68c7c8e28ba904a25efa7fcd2` |
| Sharpa left URDF and meshes | `/mnt/c/Users/Howard/dual_sharpa_wave_ros2/` | `85d02e0275e8bfb55a9402f21dba08cec7b3ab97`; its vendored model records upstream revision `0d19cac602f46456b819e4b6a2c09a74982c9a3e` |
| Display mount | `/mnt/c/Users/Howard/bimanual_manipulation/config/hand_mounts.yaml` | `left.xyz=[0,0,0]`, `left.rpy=[0,1.570796,0]`; upstream labels these mock preview placeholders |

The assembled, unmodified single-side URDF had SHA-256 `5f07f5441d04cf8b60b0df9cc92bc88c3b9fc73ec305a9cf5ff43ddb6aa2e112` when produced from these sources on 2026-10-14. The bundled `model.urdf` has SHA-256 `e7589f9918ce41bc154ae903390bbc60c696a960bc75783bf79a20fd80276e0a`. `SHA256SUMS.txt` records the 32 copied and 24 derived mesh files.

`preview.png` was rendered from the bundled URDF with MuJoCo 3.6.0 at zero joint positions, using a brighter headlight for inspection. Rendering disables geom group 0 (collision) and keeps group 1 (visual) visible. MuJoCo's default view shows both; the coarse FANUC collision meshes obscure the detailed visual meshes.
Use `python script/preview_crx_sharpa_asset.py` from the repository root for a static interactive view with the same visual-only setting. The generic MuJoCo viewer also steps physics, and this display-only model has no actuators to hold its joints against gravity.

## Preview conversion

`build_description(sides="left")` supplied the model tree. ROS `package://` mesh URIs were replaced with paths relative to this directory. The seven FANUC visual DAE scenes were loaded with `trimesh`; their 17 material regions were exported as separate transformed STL meshes in meters. Each region keeps the source DAE diffuse color as a URDF visual material. They retain the original URDF visual origins and no mesh scale. The seven original FANUC collision STL files retain their own URDF collision origins and `0.001` mesh scale. The 25 referenced Sharpa left-hand STL files were copied unchanged. The seven earlier single-color DAE-derived STL files remain in `meshes/fanuc/crx5ia/visual/` but are no longer referenced by `model.urdf`. A MuJoCo URDF compiler extension sets `strippath="false"`, `balanceinertia="true"`, and `discardvisual="false"`. Joint names, joint limits, link transforms, and the selected mount transform were retained from the assembled model.

The resulting FANUC appearance follows the source DAE geometry and diffuse colors. STL conversion does not preserve the DAE's complete shading model or any textures. Use the source ROS/RViz model to inspect those details. The display mount is not a measured hardware calibration, and this asset has not been validated for dynamics, contacts, or control.

## Attribution and permission records

The source Sharpa package records Apache-2.0, and its original license and notice are copied into `licenses/`. FANUC's source macro and package record Apache-2.0; its license text is also copied. The `bimanual_manipulation` package declares Apache-2.0 in `package.xml`. The local `dual_crx_control/package.xml` still says `TODO: License declaration`; this import does not resolve that gap or assert clearance for public redistribution of the combined asset.
