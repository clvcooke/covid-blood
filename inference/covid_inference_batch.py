from inference.covid_inference_single import run_single

if __name__ == "__main__":

    model_info = {
        '1dbrkt2i': 5,
        '2qyssv3y': 5,
        '171f08s0': 4,
        'jup6btte': 4,
        'z6u8j84u': 3,
        'jaqln5pf': 3,
        '1ngmy2jz': 2,
        '1d3hpvs0': 2,
        '1abip6we': 0,
        '3fdw8njf': 0,
        '3au2k57i': 1,
        '1l33ib88': 1
    }

    gpu_number = 0
    counter = 0
    TTA_rounds = 0
    TTA = TTA_rounds > 0
    for model_id, fold_number in model_info.items():
        if TTA:
            for TTA_round in range(TTA_rounds):
                print(f"TTA round {TTA_round + 1}/{TTA_rounds}")
                print(f"Inference for model {model_id} on fold {fold_number} -- {counter}/{len(model_info)}")
                counter += 1
                run_single(model_id, fold_number, gpu_number, TTA, TTA_round)
        else:
            print(f"Inference for model {model_id} on fold {fold_number} -- {counter}/{len(model_info)}")
            counter += 1
            run_single(model_id, fold_number, gpu_number, TTA)