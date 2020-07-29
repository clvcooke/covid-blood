from inference.covid_inference_single import run_single

if __name__ == "__main__":
    model_info = {
        '3dzierwk': 1,
    }


    model_info = {
        '3bcnioqn': 0,
        '3dzierwk': 1,
        '1629vjqj': 2,
        '29tpkvfa': 3,
        '29psm43i': 4,
        '3cqyqz7j': 5
    }

    model_info = {
        '3nlkyl8h': 0
    }

    gpu_number = 1
    counter = 0
    TTA_rounds = 5
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