from inference.covid_inference_single import run_single

if __name__ == "__main__":
    model_info = {
        '3o3dfh19': 5,
        'mh3i9q6m': 5,
        'udjgoop7': 5,
        '3mr5fvio': 4,
        'df7vkjt8': 4,
        'yxmqo2gh': 4,
        '1gjbinvl': 3,
        '9tt17wcr': 3,
        '3af6l4x5': 3,
        '1ag0y3sp': 2,
        '1nrh931e': 2,
        '2hnl1ibb': 2,
        '2hpmtsba': 1,
        '2vq060p8': 1,
        '3gocrf0d': 1,
        '1qfa3jgw': 0,
        '3p951nyk': 0,
        'q7qx2q3r': 0
    }
    gpu_number = 0
    counter = 0
    for model_id, fold_number in model_info.items():
        print(f"Inference for model {model_id} on fold {fold_number} -- {counter}/{len(model_info)}")
        counter += 1
        run_single(model_id, fold_number, gpu_number)