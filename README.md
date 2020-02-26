# aslaug3d_simulation

## Usage

`python3 train.py -s 90e6 -v some_version -p policies.aslaug_policy_v4.AslaugPolicy -f some_name -n 64`
`python3 run.py -v v12easy -cfr 0 --det 0 -f final_pt:10M --no_sleep 0 --free_cam 1`

# Evaluation in simu or on real machine

## Simulation evaluation

1. Copy the saved folder for an agent to the Raspberry Pi (rospi, use SD card wit image in Tupper ware)
2. "ros_aslaug" sources the environment. Don't forget to set rosmaster (in our case, to WGServer)

### Evaluation of the baseline

On Server:
`roslaunch aslaug_moveit_baseline full_baseline.launch moveit:=false world_name:=...`
Don't forget to unpause gazebo in GUI

On Raspberry Pi in multiple terminals:
`roslaunch aslaug_moveit_baseline moveit.launch`

`roscd aslaug_moveit_baseline; cd scripts; python perform_tests_easy.py ...`

### Evaluation of the agent

On Server:
`roslaunch aslaug_bringup aslaug_control_simu.launch rviz:=false world_name:=...`
Don't forget to unpause gazebo in GUI

On Raspberry Pi (two terminals):
`roslaunch aslaug_bringup aslaug_control_only.launch`

`roscd aslaug_brinup; cd scripts; python perform_tests.py ...`

The evaluation can be done by executing eval.py in the respective folder.

## Real machine execution

1. Startup machine and ssh to panda (for port forwarding bridge)
2. Go to https://localhost:10000/desk/ and execute script which moves panda to tests initial position
3. SSH to ridgeback and execute localization from rovioli (something like moma_demos localization.launch)
4. Execute panda velocity controller on panda (panda_control velocity.launch or so?)
5. Adapt the control launch file and the controller node to use simulation=False and launch it: roslaunch aslaug_bringup aslaug_control.launch
