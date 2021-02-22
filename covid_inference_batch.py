from inference.covid_inference_single import run_single

if __name__ == "__main__":
    model_info = {
        '3dzierwk': 1,
    }


    model_info = {
        '1zjrjiku': 0,
        '2om7nlu5': 1,
        '1sddw8kt': 2,
        '15jtunw1': 3,
        '2pskiu0r': 4,
        '3pgjwlo2': 5
    }

    gpu_number = 1
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
