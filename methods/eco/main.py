from eco.attk_model import AttackedModel
from eco.model import HFModel
from eco.classifier import PromptClassifier, TokenClassifier


llama_setup = {
    "batch_size": 32,
    "classifier_threshold": 0.999,
    "model_name": "llama2-7b",
    "corrupt_method": "zero_out_top_k",
    "corrupt_args": {"dims": 1000},
}
tofu_llama_model_path = "models/llama2-7b-tofu"

def get_eco_model(setup=llama_setup, model_path=tofu_llama_model_path):

    orig_model = HFModel(
        model_name=setup["model_name"],
        model_path=model_path,
        config_path="./config",
    )


    prompt_classifier = PromptClassifier(
        model_name="roberta-base",
        model_path="chrisliu298/tofu_forget10_classifier",
        batch_size=setup["batch_size"],
    )

    token_classifier = TokenClassifier(
        model_name="dslim/bert-base-NER",
        model_path="dslim/bert-base-NER",
        batch_size=setup["batch_size"],
    )


    model = AttackedModel(
        model=orig_model,
        prompt_classifier=prompt_classifier,
        token_classifier=token_classifier,
        corrupt_method=setup["corrupt_method"],
        corrupt_args=setup["corrupt_args"],
        classifier_threshold=setup["classifier_threshold"],
    )
    return model


    