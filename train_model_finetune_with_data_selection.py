import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from data_feed.data_feed import DataFeed
from scipy.io import savemat
from train_model import train_model


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    real_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_1"
    synth_data_root = "DeepMIMO/DeepMIMO_datasets/Boston5G_3p5_small_notree"
    train_csv = "/train_data_idx.csv"
    val_csv = "/test_data_idx.csv"
    test_csv = "/test_data_idx.csv"
    train_batch_size = 64
    test_batch_size = 1024
    num_pretrain_data = 32000

    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(1000,))

    all_all_nmse = []
    for i in range(10):
        all_nmse = []

        # pre-train on synth 32k data points
        torch.manual_seed(seeds[i])
        train_loader_ = DataLoader(
            DataFeed(
                synth_data_root,
                train_csv,
                num_data_point=num_pretrain_data,
                random_state=seeds[i],
            ),
            batch_size=train_batch_size,
            shuffle=True,
        )

        test_loader_ = DataLoader(
            DataFeed(
                real_data_root, test_csv, num_data_point=10000, random_state=seeds[i]
            ),
            batch_size=test_batch_size,
        )

        now = datetime.datetime.now().strftime("%H_%M_%S")
        date = datetime.date.today().strftime("%y_%m_%d")
        comment = "_".join([now, date])

        print("Number of trainig data points: " + str(num_pretrain_data))
        ret = train_model(
            train_loader=train_loader_,
            val_loader=None,
            test_loader=test_loader_,
            comment=comment,
            encoded_dim=32,
            num_epoch=250,
            lr=1e-2,
            if_writer=False,
            model_path=None,
            save_model=True,
        )
        model_path = ret["model_path"]

        # fine-tune on real
        for num_train_data, num_epoch in zip(
            [1000, 2000, 4000, 8000, 16000, 32000], [50, 50, 50, 50, 100, 100]
        ):
            torch.manual_seed(seeds[i])
            train_loader = DataLoader(
                DataFeed(
                    real_data_root,
                    train_csv,
                    num_data_point=num_train_data,
                    random_state=seeds[i],
                ),
                batch_size=train_batch_size,
                shuffle=True,
            )

            test_loader = DataLoader(
                DataFeed(
                    real_data_root,
                    test_csv,
                    num_data_point=10000,
                    random_state=seeds[i],
                ),
                batch_size=test_batch_size,
            )

            now = datetime.datetime.now().strftime("%H_%M_%S")
            date = datetime.date.today().strftime("%y_%m_%d")
            comment = "_".join([now, date])

            print("Number of finetuning data points : " + str(num_train_data))
            ret = train_model(
                train_loader=train_loader,
                val_loader=None,
                test_loader=test_loader,
                comment=comment,
                encoded_dim=32,
                num_epoch=num_epoch,
                if_writer=False,
                model_path=model_path,
                lr=1e-4,
            )
            all_nmse.append(ret["test_nmse"])
        all_nmse = np.asarray(all_nmse)
        all_all_nmse.append(all_nmse)
    all_all_nmse = np.stack(all_all_nmse, 0)

    print(all_all_nmse)
    savemat(
        "result4/all_nmse_finetune.mat",
        {"all_nmse_finetune": all_all_nmse},
    )
    print("done")
