import subprocess

# 定义参数列表，使用不同的 batch_size, num_epochs, learning_rate, drop_rate

batch_sizes = [64, 128, 256]
learning_rates = [1e-2, 1e-3, 1e-4]
drop_rates = [0.3, 0.5, 0.7]

# for inverted images
# batch_sizes = [128]
# learning_rates = [1e-3 * 1.5]
# drop_rates = [0]

params_list = []
for item in batch_sizes:
    for item2 in learning_rates:
        for item3 in drop_rates:
            params_list.append({"batch_size": item, "learning_rate": item2, "drop_rate": item3, "num_epochs": 50})

# 遍历参数列表，并为每组参数执行训练
for params in params_list:
    # 构建命令
    command = (
        f"python main.py "
        f"--batch_size {params['batch_size']} "
        f"--num_epochs {params['num_epochs']} "
        f"--learning_rate {params['learning_rate']} "
        f"--drop_rate {params['drop_rate']} "
        f"--is_train True "
        f"--data_dir ../cifar-10_data "
        f"--train_dir ./train "
        f"--struct_order 114514"
    )

    # 打印当前执行的命令
    print(f"Running command: {command}")
    
    # 使用 subprocess 运行命令
    subprocess.run(command, shell=True)
    
    print(f"Finished running: {command}\n")
