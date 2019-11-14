import os

os.system("module load python/3.6.0")
os.system("module load open_mpi")

for i in range(1, 129):
    cmd = "bsub -n 16 -R \"rusage[mem=2000]\" -W 9:00 \"python euler_train_random.py -s 15e6 -v v2 -p policies.aslaug_policy_v1hw.AslaugPolicy -f v1_euler_{}\"".format(i)
    os.system(cmd)
    print("Submitted job {}.".format(i))
