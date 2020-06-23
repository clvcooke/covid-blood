from inference.covid_inference_single import run_single

if __name__ == "__main__":
    model_ids = ['21y6wxsf', '17kqtr5n', '2apsqyhl', '3jsbms28', 'vdpbmwg3', '3fmjh08j']
    fold_numbers = [5, 4, 3, 2, 1, 0]
    gpu_number = 0
    for model_id, fold_number in zip(model_ids, fold_numbers):
        print(f"Inference for model {model_id} on fold {fold_number}")
        run_single(model_id, fold_number, gpu_number)