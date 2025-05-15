import os
import sys
import logging
import argparse
import pandas as pd
import datetime
import datasets

from utils import (
    execute_tests,
    extract_code_from_backticks,
)

from config import get_model_module, ROOT_DIR


def setup_logging(args, model_id):
    """Set up logging to both console and file."""
    # Get log directory from environment variable if set by SLURM script
    curr_date_format = datetime.datetime.now().strftime("%Y-%m-%d")
    curr_hour_minute_format = datetime.datetime.now().strftime("%H-%M")
    
    log_dir = f"{ROOT_DIR}/launcher_logs/{curr_date_format}/multi_turn_{args.interaction}/{curr_hour_minute_format}_{model_id}/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/detailed_log.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main():
    ## setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interaction",
        type=str,
        default="refactor",
        help="The interaction to evaluate",
        choices=["expansion", "editing", "refactor"],
    )
    parser.add_argument(
        "--model_name", type=str, default="gpt-4o", help="The model to use"
    )
    parser.add_argument("--model_temp", type=float, default=0.0)
    args = parser.parse_args()
    
        ## Load the model
    model_module = get_model_module(args.model_name)
    model_dict = model_module.load_model(args)

    if len(args.model_name.split("/")) > 1:
        model_id = (
            args.model_name.split("/")[-1]
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )
    else:
        model_id = (
            args.model_name.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )

    logger = setup_logging(args, model_id)
    logger.info(f"Extracted Model ID: {model_id}")
    
    mt_sec_interaction = datasets.load_dataset("ai-sec-lab/mt-sec", split=args.interaction).to_pandas()
        
    save_dir = f"{ROOT_DIR}/results/multi_turn/{model_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{args.interaction}_multi_turn"

    logger.info(f"Dataset Size: {len(mt_sec_interaction)}")
    logger.info(f"Save Path: {save_path}")

    if os.path.isfile(save_path + ".pkl") and not args.overwrite:
        logger.info(f"Loading the checkpoint from {save_path}")
        mt_sec_interaction = pd.read_pickle(save_path + ".pkl")
    else:
        # Add columns to store the results
        mt_sec_interaction["multi_turn_responses"] = None  ## list of multi_turn responses
        mt_sec_interaction["multi_turn_responses_last"] = None  ## the last multi_turn response
        mt_sec_interaction["multi_turn_responses_last_code"] = (
            None  ## extracted code from the last multi_turn response
        )
        mt_sec_interaction["result_capability"] = None
        mt_sec_interaction["result_safety"] = None
        mt_sec_interaction["reasoning_chain"] = None

    count = 0
    for row_idx, data_row in mt_sec_interaction.iterrows():
        assert mt_sec_interaction.at[row_idx, "id"] == data_row["id"]

        logger.info("=" * 50)                
        logger.info(f"Processing row {row_idx} [{count}/{len(mt_sec_interaction)}]")
        logger.info("=" * 50)

        turns = data_row["generated_multi_turns"]
        try:
            assert len(turns) == 3
        except:
            continue

        for turn_idx in range(3):
            turns[
                turn_idx
            ] += "\nOnly generate the Python function (and any imported packages), not the example usage code, etc.\n"

        try:
            response_block = model_module.get_multi_model_response(
                model_dict, turns
            )
        except Exception as e:
            logger.info(f"Error in getting response: {e}")
            continue

        final_multi_turn_response, all_multi_turn_responses, *extra = response_block
        reasoning_chains = extra[0] if extra else None

        ### extract code from backticks
        final_multi_turn_response_code = extract_code_from_backticks(
            final_multi_turn_response
        )
        final_multi_turn_response_code = final_multi_turn_response_code.removeprefix(
            "```python"
        )
        final_multi_turn_response_code = final_multi_turn_response_code.removeprefix(
            "python"
        )
        final_multi_turn_response_code = final_multi_turn_response_code.removesuffix(
            "```"
        )
        final_multi_turn_response_code = final_multi_turn_response_code.removesuffix(
            "python"
        )

        ## construct the full script
        all_multi_turn_codes = ""
        for turn_res in all_multi_turn_responses:
            turn_res_code = extract_code_from_backticks(turn_res)

            ## failsafe
            turn_res_code = turn_res_code.removeprefix("```python")
            turn_res_code = turn_res_code.removeprefix("python")
            turn_res_code = turn_res_code.removesuffix("```")
            turn_res_code = turn_res_code.removesuffix("python")
            all_multi_turn_codes += turn_res_code + "\n\n"

        
        ## Evaluate the results
        multi_turns_eval = execute_tests(
            code=all_multi_turn_codes,
            setup=data_row["unittest"]["setup"],
            testcases=data_row["unittest"]["testcases"],
            func_name=data_row["task_description"]["function_name"],
            install_requires=data_row["install_requires"],
        )
        logger.info(f"Multi-turns evaluation: {multi_turns_eval}")

        ## Save the results in the dataframe itself
        mt_sec_interaction.at[row_idx, "multi_turn_responses"] = all_multi_turn_responses
        mt_sec_interaction.at[row_idx, "multi_turn_responses_last"] = final_multi_turn_response
        mt_sec_interaction.at[row_idx, "multi_turn_responses_last_code"] = (
            final_multi_turn_response_code
        )
        mt_sec_interaction.at[row_idx, "reasoning_chain"] = reasoning_chains

        if multi_turns_eval is None:
            logger.info(f"Error: Could not evaluate the code for row {row_idx}")
            logger.info(f"Got: {multi_turns_eval}")
            continue

        mt_sec_interaction.at[row_idx, "result_capability"] = multi_turns_eval[
            "capability"
        ]
        mt_sec_interaction.at[row_idx, "result_safety"] = multi_turns_eval["safety"]

        count += 1
        if count % 20 == 0:
            logger.info("=-" * 50)
            logger.info(f"Saving the dataframe at row {row_idx}")
            logger.info("=-" * 50)

            ## save the dataframe
            mt_sec_interaction.to_pickle(f"{save_path}.pkl")
            mt_sec_interaction.to_csv(f"{save_path}.csv")

    logger.info("=-" * 50)
    logger.info("Finished processing all rows")
    logger.info("=-" * 50)
    ## save the dataframe
    mt_sec_interaction.to_pickle(f"{save_path}.pkl")
    mt_sec_interaction.to_csv(f"{save_path}.csv")

        
    

if __name__ == "__main__":
    main()