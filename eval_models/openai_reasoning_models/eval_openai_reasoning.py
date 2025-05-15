from openai import OpenAI
import os
import argparse

OPENAI_API_KEY = os.getenv("OPENAI_KEY")


def openai_completion(client, model_id, conversation_log):

    response = client.chat.completions.create(model=model_id, messages=conversation_log)

    conversation_log.append(
        {
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content.strip(),
        }
    )
    return conversation_log


def load_model(args):
    client = OpenAI(api_key=OPENAI_API_KEY)
    return {
        "openai_client": client,
        "openai_model": args.model_name,
    }


def get_multi_model_response(model_dict, prompts, verbose=False, system_prompt=None):

    openai_client, openai_model = (
        model_dict["openai_client"],
        model_dict["openai_model"],
    )

    if system_prompt is None:
        conversations = [
            {
                "role": "developer",
                "content": "You are a helpful assistant.",
            },
        ]
    else:
        conversations = [
            {
                "role": "developer",
                "content": f"You are a helpful assistant. {system_prompt}",
            },
        ]
    model_responses = []

    for prompt in prompts:
        prompt_conv = {"role": "user", "content": prompt}
        conversations.append(prompt_conv)

        conversations = openai_completion(
            client=openai_client,
            model_id=openai_model,
            conversation_log=conversations,
        )

        if verbose:
            print(f"Running Conversation: {conversations}")
            print()

        model_responses.append(conversations[-1]["content"].strip())
        
    return model_responses[-1], model_responses, None, conversations


def get_single_model_response(model_dict, prompt):

    openai_client, openai_model = (
        model_dict["openai_client"],
        model_dict["openai_model"],
    )

    prompt_conv = {"role": "user", "content": prompt}
    conversations = [prompt_conv]

    conversations = openai_completion(
        client=openai_client,
        model_id=openai_model,
        conversation_log=conversations,
    )

    return conversations[-1]["content"].strip(), None


def get_single_model_response_diff(model_dict, conversations):

    openai_client, openai_model = (
        model_dict["openai_client"],
        model_dict["openai_model"],
    )

    conversations = openai_completion(
        client=openai_client,
        model_id=openai_model,
        conversation_log=conversations,
    )

    return conversations[-1]["content"].strip(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="o1-mini-2024-09-12",
    )
    args = parser.parse_args()

    model_dict = load_model(args)

    ## test single turn
    response = get_single_model_response(model_dict, "What is the capital of France?")
    print("-" * 50)
    print(f"Single-Turn Response:")
    print(response)
    print("-" * 50)

    ## test multi turn
    prompts = [
        "Write a paragraph on the topic of 'The importance of education'.",
        "Now summarize the above paragraph in one lines.",
    ]
    last_response, all_responses = get_multi_model_response(
        model_dict, prompts, verbose=True
    )

    print("-" * 50)
    print(f"Multi-Turn Response:")
    print(last_response)
    print(all_responses)
    print("-" * 50)


if __name__ == "__main__":
    main()
