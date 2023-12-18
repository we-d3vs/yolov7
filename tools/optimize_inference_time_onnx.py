import argparse
import time
import numpy as np
# https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
# import torch
import onnx
import onnxruntime as ort
from typing import Dict, List
from speedster import optimize_model, save_model, load_model


def get_input_names(onnx_model: str):
    model = onnx.load(onnx_model)
    input_all = [node.name for node in model.graph.input]
    return input_all


def get_output_names(onnx_model: str):
    model = onnx.load(onnx_model)
    output_all = [node.name for node in model.graph.output]
    return output_all


def run_onnx_model(
    onnx_model: str,
    session: ort.InferenceSession, input_tensors: List[np.ndarray], inputs: Dict, output_names: str
) -> List[np.ndarray]:
    
    res = session.run(
        output_names=output_names, input_feed=inputs
    )
    return list(res)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Optimize ONNX model for inference time with Speedster."
                                     " Make sure that you did the export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH_TO/tensorrt")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model")
    parser.add_argument("--optimized_model", type=str, required=True, help="Path to the optimized ONNX model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the model. Otherwise, optimize the model.")

    args = parser.parse_args()

    model = args.model
    optimize_model_name = args.optimized_model

    if not args.benchmark:
        input_data = [((np.random.randn(1, 3, 640, 640).astype(np.float32), ), np.array([0])) for i in range(100)]

        # Run Speedster optimization
        optimized_model = optimize_model(
            model, input_data=input_data, 
            optimization_time="unconstrained",
            metric_drop_ths=0.01,
            store_latencies=True
        )

        save_model(optimized_model, optimize_model_name)

    session = ort.InferenceSession(
        model,
        providers=["CUDAExecutionProvider"] # Change to ["CPUExecutionProvider"] if run on cpu
    )

    x = np.random.randn(1, 3, 640, 640).astype(np.float32)

    inputs = {
        name: array
        for name, array in zip(get_input_names(model), [x])
    }

    res_original = run_onnx_model(model, session, [x], inputs, get_output_names(model))

    num_iters = 100

    # Warmup
    for i in range(10):
        run_onnx_model(model, session, [x], inputs, get_output_names(model))

    start = time.time()
    for i in range(num_iters):
        run_onnx_model(model, session, [x], inputs, get_output_names(model))
    stop = time.time()

    print("Average latency original model: {:.4f} seconds".format((stop - start) / num_iters))


    # Finally we compute the average latency for the optimized model:
    optimized_model = load_model(optimize_model_name)
    for i in range(10):
        optimized_model(x)

    start = time.time()
    for i in range(num_iters):
        optimized_model(x)
    stop = time.time()

    print("Average latency optimized model: {:.4f} seconds".format((stop - start) / num_iters))