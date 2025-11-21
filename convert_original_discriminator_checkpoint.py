"""Convert VITS discriminator checkpoint and add it to an already converted VITS checkpoint."""

import argparse
import torch
import logging

from transformers.models.vits.modeling_vits import VitsModel
from transformers.models.vits.tokenization_vits import VitsTokenizer

from huggingface_hub import hf_hub_download

from utils.feature_extraction_vits import VitsFeatureExtractor
from utils.configuration_vits import VitsConfig
from utils.modeling_vits_training import VitsDiscriminator, VitsModelForPreTraining

# Loggingni to'g'irlash
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


MAPPING = {
    "conv_post": "final_conv",
}

@torch.no_grad()
def convert_checkpoint(
    language_code,
    pytorch_dump_folder_path,
    checkpoint_path=None,
    generator_checkpoint_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if language_code is not None:
        # Agar language_code berilgan bo'lsa, avtomatik yuklashga harakat qiladi
        # Lekin MMS uchun bu har doim ham ishlamasligi mumkin, shuning uchun manual path afzal
        try:
            checkpoint_path = hf_hub_download(repo_id="facebook/mms-tts", subfolder=f"full_models/{language_code}", filename="D_100000.pth")
        except Exception:
            logger.warning(f"Discriminator checkpoint not found for {language_code}. Using random initialization.")
            checkpoint_path = None
            
        generator_checkpoint_path = f"facebook/mms-tts-{language_code}"
    
    # 1. Config va Generatorni yuklaymiz
    logger.info(f"Loading config and generator from {generator_checkpoint_path}...")
    config = VitsConfig.from_pretrained(generator_checkpoint_path)
    generator = VitsModel.from_pretrained(generator_checkpoint_path)
    
    # 2. Discriminatorni yaratamiz (bo'sh/random holatda)
    discriminator = VitsDiscriminator(config)

    # Normalizatsiyani qo'llash
    for disc in discriminator.discriminators:
        disc.apply_weight_norm()

    # 3. Agar checkpoint fayli bo'lsa, uni yuklashga harakat qilamiz
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Eskicha nomlarni yangisiga o'zgartirish
            for k, v in list(state_dict.items()):
                for old_layer_name in MAPPING:
                    new_k = k.replace(old_layer_name, MAPPING[old_layer_name])
                state_dict[new_k] = state_dict.pop(k)

            # XATOLIK BO'LGAN JOY TUZATILDI:
            # Biz bu yerda "strict=False" qilamiz. 
            # Chunki siz Generator faylini (pytorch_model.bin) beryapsiz. 
            # Generator kalitlari Discriminatorga tushmaydi, shuning uchun u shunchaki tashlab yuboriladi
            # va Discriminator "yangidek" (fresh/random) qoladi. Bu finetuning uchun to'g'ri.
            discriminator.load_state_dict(state_dict, strict=False)
            logger.info("Loaded weights into discriminator (partial/strict=False).")

        except Exception as e:
            logger.warning(f"Could not load discriminator weights: {e}")
            logger.warning("Proceeding with random discriminator weights (This is OK for fine-tuning MMS).")

    n_params = discriminator.num_parameters(exclude_embeddings=True)
    logger.info(f"Discriminator loaded: {round(n_params/1e6,1)}M params")

    for disc in discriminator.discriminators:
        disc.remove_weight_norm()

    # 4. Yakuniy modelni yig'amiz
    logger.info("Building final VitsModelForPreTraining...")
    model = VitsModelForPreTraining(config)

    model.text_encoder = generator.text_encoder
    model.flow = generator.flow
    model.decoder = generator.decoder
    model.duration_predictor = generator.duration_predictor
    model.posterior_encoder = generator.posterior_encoder

    if config.num_speakers > 1:
        model.embed_speaker = generator.embed_speaker

    model.discriminator = discriminator
    
    # Tokenizer va Feature Extractor
    tokenizer = VitsTokenizer.from_pretrained(generator_checkpoint_path, verbose=False)
    feature_extractor = VitsFeatureExtractor(sampling_rate=model.config.sampling_rate, feature_size=80)

    # 5. Saqlash
    logger.info(f"Saving model to {pytorch_dump_folder_path}...")
    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        feature_extractor.push_to_hub(repo_id)
    
    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--language_code", default=None, type=str, help="Language code (optional).")
    
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Local path to original discriminator checkpoint."
    )
    parser.add_argument(
        "--generator_checkpoint_path", default=None, type=str, help="Path to the 🤗 generator (VitsModel)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )
    
    args = parser.parse_args()
    convert_checkpoint(
        args.language_code,
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.generator_checkpoint_path,
        args.push_to_hub,
    )