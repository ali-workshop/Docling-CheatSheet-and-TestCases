# Docling Cheat Sheet 📚

This comprehensive cheat sheet provides a quick reference for using Docling, covering various conversion methods, serialization, chunking, and RAG with AI development frameworks. Each section includes practical examples and code snippets to help you get started quickly. ✨

## 🔀 Conversion

### Simple conversion

```python
from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869" # document per local path or URL

converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())
# output: ## Docling Technical Report [...]
```



### Custom conversion

This example demonstrates how to customize the conversion pipeline, including enabling OCR, table structure detection, and specifying OCR language and accelerator options.

```python
import json
import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    # Docling Parse with EasyOCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    ## Export results
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_result.document.export_to_dict()))

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_document_tokens())

if __name__ == "__main__":
    main()
```



### Batch conversion

This example shows how to convert multiple documents in a batch and export them to various formats like JSON, HTML, Markdown, and plain text.

```python
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

import yaml
from docling_core.types.doc import ImageRefMode

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

USE_V2 = True
USE_LEGACY = False

def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_html(
                    output_dir / f"{doc_filename}.html",
                    image_mode=ImageRefMode.EMBEDDED,
                )
                conv_res.document.save_as_document_tokens(
                    output_dir / f"{doc_filename}.doctags.txt"
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.txt",
                    image_mode=ImageRefMode.PLACEHOLDER,
                    strict_text=True,
                )

                # Export Docling document format to YAML:
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))

                # Export Docling document format to doctags:
                with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_document_tokens())

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            if USE_LEGACY:
                # Export Deep Search document JSON format:
                with (output_dir / f"{doc_filename}.legacy.json").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))

                # Export Text format:
                with (output_dir / f"{doc_filename}.legacy.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(
                        conv_res.legacy_document.export_to_markdown(strict_text=True)
                    )

                # Export Markdown format:
                with (output_dir / f"{doc_filename}.legacy.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())

                # Export Document Tags format:
                with (output_dir / f"{doc_filename}.legacy.doctags.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.legacy_document.export_to_document_tokens())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_paths = [
        data_folder / "pdf/2206.01062.pdf",
        data_folder / "pdf/2203.01017v2.pdf",
        data_folder / "pdf/2305.03393v1.pdf",
        data_folder / "pdf/redp5110_sampled.pdf",
    ]

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=DoclingParseV4DocumentBackend
            )
        }
    )

    start_time = time.time()

    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=Path("scratch")
    )

    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(input_doc_paths)}."
        )

if __name__ == "__main__":
    main()
```



### Multi-format conversion

This example demonstrates how to configure Docling to handle multiple input formats (PDF, HTML, DOCX, PPTX, etc.) and convert them into a unified document representation.

```python
import json
import logging
from pathlib import Path

import yaml

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

_log = logging.getLogger(__name__)

def main():
    input_paths = [
        Path("README.md"),
        Path("tests/data/html/wiki_duck.html"),
        Path("tests/data/docx/word_sample.docx"),
        Path("tests/data/docx/lorem_ipsum.docx"),
        Path("tests/data/pptx/powerpoint_sample.pptx"),
        Path("tests/data/2305.03393v1-pg9-img.png"),
        Path("tests/data/pdf/2206.01062.pdf"),
        Path("tests/data/asciidoc/test_01.asciidoc"),
    ]

    ## to customize use:

    doc_converter = (
        DocumentConverter(  # all of the below is optional, has internal defaults.
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.MD,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, backend=PyPdfiumDocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline  # , backend=MsWordDocumentBackend
                ),
            },
        )
    )

    conv_results = doc_converter.convert_all(input_paths)

    for res in conv_results:
        out_path = Path("scratch")
        print(
            f"Document {res.input.file.name} converted."
            f"\nSaved markdown output to: {out_path!s}"
        )
        _log.debug(res.document._export_to_indented_text(max_text_len=16))
        # Export Docling document format to markdowndoc:
        with (out_path / f"{res.input.file.stem}.md").open("w") as fp:
            fp.write(res.document.export_to_markdown())

        with (out_path / f"{res.input.file.stem}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        with (out_path / f"{res.input.file.stem}.yaml").open("w") as fp:
            fp.write(yaml.safe_dump(res.document.export_to_dict()))

if __name__ == "__main__":
    main()
```



### VLM pipeline with SmolDocling

This example shows how to use the VLM pipeline with SmolDocling, a compact vision-language model, for document conversion.

```python
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

source = "https://arxiv.org/pdf/2501.17887"

##### USING SIMPLE DEFAULT VALUES

# SmolDocling model
# Using the transformers framework

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())

##### USING MACOS MPS ACCELERATOR

# For more options see the compare_vlm_models.py example.

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_MLX,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
```



### VLM pipeline with remote model

This example demonstrates how to use the VLM pipeline with remote models, including configurations for LM Studio, Ollama, and IBM watsonx.ai.

```python
import logging
import os
from pathlib import Path
from typing import Optional

import requests
from docling_core.types.doc.page import SegmentedPage
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

## Example of ApiVlmOptions definitions

### Using LM Studio

def lms_vlm_options(model: str, prompt: str, format: ResponseFormat):
    options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",  # the default LM Studio
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=format,
    )
    return options

### Using LM Studio with OlmOcr model

def lms_olmocr_vlm_options(model: str):
    def _dynamic_olmocr_prompt(page: Optional[SegmentedPage]):
        if page is None:
            return (
                "Below is the image of one page of a document. Just return the plain text"
                " representation of this document as if you were reading it naturally.\n"
                "Do not hallucinate.\n"
            )

        anchor = [
            f"Page dimensions: {int(page.dimension.width)}x{int(page.dimension.height)}"
        ]

        for text_cell in page.textline_cells:
            if not text_cell.text.strip():
                continue
            bbox = text_cell.rect.to_bounding_box().to_bottom_left_origin(
                page.dimension.height
            )
            anchor.append(f"[{int(bbox.l)}x{int(bbox.b)}] {text_cell.text}")

        for image_cell in page.bitmap_resources:
            bbox = image_cell.rect.to_bounding_box().to_bottom_left_origin(
                page.dimension.height
            )
            anchor.append(
                f"[Image {int(bbox.l)}x{int(bbox.b)} to {int(bbox.r)}x{int(bbox.t)}]"
            )

        if len(anchor) == 1:
            anchor.append(
                f"[Image 0x0 to {int(page.dimension.width)}x{int(page.dimension.height)}]"
            )

        # Original prompt uses cells sorting. We are skipping it in this demo.

        base_text = "\n".join(anchor)

        return (
            f"Below is the image of one page of a document, as well as some raw textual"
            f" content that was previously extracted for it. Just return the plain text"
            f" representation of this document as if you were reading it naturally.\n"
            f"Do not hallucinate.\n"
            f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
        )

    options = ApiVlmOptions(
        url="http://localhost:1234/v1/chat/completions",
        params=dict(
            model=model,
        ),
        prompt=_dynamic_olmocr_prompt,
        timeout=90,
        scale=1.0,
        max_size=1024,  # from OlmOcr pipeline
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

### Using Ollama

def ollama_vlm_options(model: str, prompt: str):
    options = ApiVlmOptions(
        url="http://localhost:11434/v1/chat/completions",  # the default Ollama endpoint
        params=dict(
            model=model,
        ),
        prompt=prompt,
        timeout=90,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

### Using a cloud service like IBM watsonx.ai

def watsonx_vlm_options(model: str, prompt: str):
    load_dotenv()
    api_key = os.environ.get("WX_API_KEY")
    project_id = os.environ.get("WX_PROJECT_ID")

    def _get_iam_access_token(api_key: str) -> str:
        res = requests.post(
            url="https://iam.cloud.ibm.com/identity/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={api_key}",
        )
        res.raise_for_status()
        api_out = res.json()
        print(f"{api_out=}")
        return api_out["access_token"]

    options = ApiVlmOptions(
        url="https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29",
        params=dict(
            model_id=model,
            project_id=project_id,
            parameters=dict(
                max_new_tokens=400,
            ),
        ),
        headers={
            "Authorization": "Bearer " + _get_iam_access_token(api_key=api_key),
        },
        prompt=prompt,
        timeout=60,
        response_format=ResponseFormat.MARKDOWN,
    )
    return options

## Usage and conversion

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2305.03393v1-pg9.pdf"

    pipeline_options = VlmPipelineOptions(
        enable_remote_services=True  # <-- this is required!
    )

    # The ApiVlmOptions() allows to interface with APIs supporting
    # the multi-modal chat interface. Here follow a few example on how to configure those.

    # One possibility is self-hosting model, e.g. via LM Studio, Ollama or others.

    # Example using the SmolDocling model with LM Studio:
    # (uncomment the following lines)
    pipeline_options.vlm_options = lms_vlm_options(
        model="smoldocling-256m-preview-mlx-docling-snap",
        prompt="Convert this page to docling.",
        format=ResponseFormat.DOCTAGS,
    )

    # Example using the Granite Vision model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_vlm_options(
    #     model="granite-vision-3.2-2b",
    #     prompt="OCR the full page to markdown.",
    #     format=ResponseFormat.MARKDOWN,
    # )

    # Example using the OlmOcr (dynamic prompt) model with LM Studio:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = lms_olmocr_vlm_options(
    #     model="hf.co/lmstudio-community/olmOCR-7B-0225-preview-GGUF",
    # )

    # Example using the Granite Vision model with Ollama:
    # (uncomment the following lines)
    # pipeline_options.vlm_options = ollama_vlm_options(
    #     model="granite3.2-vision:2b",
    #     prompt="OCR the full page to markdown.",
    # )

    # Another possibility is using online services, e.g. watsonx.ai.
    # Using requires setting the env variables WX_API_KEY and WX_PROJECT_ID.
    # (uncomment the following lines)
    # pipeline_options.vlm_options = watsonx_vlm_options(
    #     model="ibm/granite-vision-3-2-2b", prompt="OCR the full page to markdown."
    # )

    # Create the DocumentConverter and launch the conversion.
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )
    result = doc_converter.convert(input_doc_path)
    print(result.document.export_to_markdown())

if __name__ == "__main__":
    main()
```



### Compare VLM models

This example compares the runtime and output quality of different vision-language models within the VLM pipeline.

```python
import json
import sys
import time
from pathlib import Path

from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from tabulate import tabulate

from docling.datamodel import vlm_model_specs
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
    TransformersModelType,
    TransformersPromptStyle,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

def convert(sources: list[Path], converter: DocumentConverter):
    model_id = pipeline_options.vlm_options.repo_id.replace("/", "_")
    framework = pipeline_options.vlm_options.inference_framework
    for source in sources:
        print("================================================")
        print("Processing...")
        print(f"Source: {source}")
        print("---")
        print(f"Model: {model_id}")
        print(f"Framework: {framework}")
        print("================================================")
        print("")

        res = converter.convert(source)

        print("")

        fname = f"{res.input.file.stem}-{model_id}-{framework}"

        inference_time = 0.0
        for i, page in enumerate(res.pages):
            inference_time += page.predictions.vlm_response.generation_time
            print("")
            print(
                f" ---------- Predicted page {i} in {pipeline_options.vlm_options.response_format} in {page.predictions.vlm_response.generation_time} [sec]:"
            )
            print(page.predictions.vlm_response.text)
            print(" ---------- ")

        print("===== Final output of the converted document =======")

        with (out_path / f"{fname}.json").open("w") as fp:
            fp.write(json.dumps(res.document.export_to_dict()))

        res.document.save_as_json(
            out_path / f"{fname}.json",
            image_mode=ImageRefMode.PLACEHOLDER,
        )
        print(f" => produced {out_path / fname}.json")

        res.document.save_as_markdown(
            out_path / f"{fname}.md",
            image_mode=ImageRefMode.PLACEHOLDER,
        )
        print(f" => produced {out_path / fname}.md")

        res.document.save_as_html(
            out_path / f"{fname}.html",
            image_mode=ImageRefMode.EMBEDDED,
            labels=[*DEFAULT_EXPORT_LABELS, DocItemLabel.FOOTNOTE],
            split_page_view=True,
        )
        print(f" => produced {out_path / fname}.html")

        pg_num = res.document.num_pages()
        print("")
        print(
            f"Total document prediction time: {inference_time:.2f} seconds, pages: {pg_num}"
        )
        print("====================================================")

        return [
            source,
            model_id,
            str(framework),
            pg_num,
            inference_time,
        ]

if __name__ == "__main__":
    sources = [
        "tests/data/pdf/2305.03393v1-pg9.pdf",
    ]

    out_path = Path("scratch")
    out_path.mkdir(parents=True, exist_ok=True)

    ## Definiton of more inline models
    llava_qwen = InlineVlmOptions(
        repo_id="llava-hf/llava-interleave-qwen-0.5b-hf",
        # prompt="Read text in the image.",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!",
        # prompt="Parse the reading order of this document.",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU],
        scale=2.0,
        temperature=0.0,
    )

    # Note that this is not the expected way of using the Dolphin model, but it shows the usage of a raw prompt.
    dolphin_oneshot = InlineVlmOptions(
        repo_id="ByteDance/Dolphin",
        prompt="<s>Read text in the image. <Answer/>",
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_IMAGETEXTTOTEXT,
        transformers_prompt_style=TransformersPromptStyle.RAW,
        supported_devices=[AcceleratorDevice.CUDA, AcceleratorDevice.CPU],
        scale=2.0,
        temperature=0.0,
    )

    ## Use VlmPipeline
    pipeline_options = VlmPipelineOptions()
    pipeline_options.generate_page_images = True

    ## On GPU systems, enable flash_attention_2 with CUDA:
    # pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
    # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

    vlm_models = [
        ## DocTags / SmolDocling models
        vlm_model_specs.SMOLDOCLING_MLX,
        vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
        ## Markdown models (using MLX framework)
        vlm_model_specs.QWEN25_VL_3B_MLX,
        vlm_model_specs.PIXTRAL_12B_MLX,
        vlm_model_specs.GEMMA3_12B_MLX,
        ## Markdown models (using Transformers framework)
        vlm_model_specs.GRANITE_VISION_TRANSFORMERS,
        vlm_model_specs.PHI4_TRANSFORMERS,
        vlm_model_specs.PIXTRAL_12B_TRANSFORMERS,
        ## More inline models
        dolphin_oneshot,
        llava_qwen,
    ]

    # Remove MLX models if not on Mac
    if sys.platform != "darwin":
        vlm_models = [
            m for m in vlm_models if m.inference_framework != InferenceFramework.MLX
        ]

    rows = []
    for vlm_options in vlm_models:
        pipeline_options.vlm_options = vlm_options

        ## Set up pipeline for PDF or image inputs
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            },
        )

        row = convert(sources=sources, converter=converter)
        rows.append(row)

        print(
            tabulate(
                rows, headers=["source", "model_id", "framework", "num_pages", "time"]
            )
        )

        print("see if memory gets released ...")
        time.sleep(10)
```



### ASR pipeline with Whisper

This example demonstrates how to use Docling's ASR (Automatic Speech Recognition) pipeline with the Whisper model to convert audio files into text.

```python
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from docling.datamodel import asr_model_specs
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline

def get_asr_converter():
    """Create a DocumentConverter configured for ASR with whisper_turbo model."""
    pipeline_options = AsrPipelineOptions()
    pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter

def asr_pipeline_conversion(audio_path: Path) -> DoclingDocument:
    """ASR pipeline conversion using whisper_turbo"""
    # Check if the test audio file exists
    assert audio_path.exists(), f"Test audio file not found: {audio_path}"

    converter = get_asr_converter()

    # Convert the audio file
    result: ConversionResult = converter.convert(audio_path)

    # Verify conversion was successful
    assert result.status == ConversionStatus.SUCCESS, (
        f"Conversion failed with status: {result.status}"
    )
    return result.document

if __name__ == "__main__":
    audio_path = Path("tests/data/audio/sample_10s.mp3")

    doc = asr_pipeline_conversion(audio_path=audio_path)
    print(doc.export_to_markdown())

    # Expected output:
    #
    # [time: 0.0-4.0]  Shakespeare on Scenery by Oscar Wilde
    #
    # [time: 5.28-9.96]  This is a LibriVox recording. All LibriVox recordings are in the public domain.
```



### Figure export

This example demonstrates how to export figures from documents, including page images, figures, and tables, and save them in various formats.

```python
import logging
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    # Important: For operating with page images, we must keep them, otherwise the DocumentConverter
    # will destroy them for cleaning up memory.
    # This is done by setting PdfPipelineOptions.images_scale, which also defines the scale of images.
    # scale=1 correspond of a standard 72 DPI image
    # The PdfPipelineOptions.generate_* are the selectors for the document elements which will be enriched
    # with the image field
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    end_time = time.time() - start_time

    _log.info(f"Document converted and figures exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
```



### Table export

This example demonstrates how to extract and export tables from documents into various formats like DataFrames, CSV, and HTML.

```python
import logging
import time
from pathlib import Path

import pandas as pd

from docling.document_converter import DocumentConverter

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    doc_converter = DocumentConverter()

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_df.to_csv(element_csv_filename)

        # Save the table as html
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

    end_time = time.time() - start_time

    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
```



### Multimodal export

This example shows how to perform multimodal export, generating dataframes with document content, page images, and other metadata.

```python
import datetime
import logging
import time
from pathlib import Path

import pandas as pd

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.utils.export import generate_multimodal_pages
from docling.utils.utils import create_hash

_log = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    # Important: For operating with page images, we must keep them, otherwise the DocumentConverter
    # will destroy them for cleaning up memory.
    # This is done by setting AssembleOptions.images_scale, which also defines the scale of images.
    # scale=1 correspond of a standard 72 DPI image
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (
        content_text,
        content_md,
        content_dt,
        page_cells,
        page_segments,
        page,
    ) in generate_multimodal_pages(conv_res):
        dpi = page._default_image_scale * 72

        rows.append(
            {
                "document": conv_res.input.file.name,
                "hash": conv_res.input.document_hash,
                "page_hash": create_hash(
                    conv_res.input.document_hash + ":" + str(page.page_no - 1)
                ),
                "image": {
                    "width": page.image.width,
                    "height": page.image.height,
                    "bytes": page.image.tobytes(),
                },
                "cells": page_cells,
                "contents": content_text,
                "contents_md": content_md,
                "contents_dt": content_dt,
                "segments": page_segments,
                "extra": {
                    "page_num": page.page_no + 1,
                    "width_in_points": page.size.width,
                    "height_in_points": page.size.height,
                    "dpi": dpi,
                },
            }
        )

    # Generate one parquet from all documents
    df_result = pd.json_normalize(rows)
    now = datetime.datetime.now()
    output_filename = output_dir / f"multimodal_{now:%Y-%m-%d_%H%M%S}.parquet"
    df_result.to_parquet(output_filename)

    end_time = time.time() - start_time

    _log.info(
        f"Document converted and multimodal pages generated in {end_time:.2f} seconds."
    )

    # This block demonstrates how the file can be opened with the HF datasets library
    # from datasets import Dataset
    # from PIL import Image
    # multimodal_df = pd.read_parquet(output_filename)

    # # Convert pandas DataFrame to Hugging Face Dataset and load bytes into image
    # dataset = Dataset.from_pandas(multimodal_df)
    # def transforms(examples):
    #     examples["image"] = Image.frombytes("RGB", (examples["image.width"], examples["image.height"]), examples["image.bytes"], "raw")
    #     return examples
    # dataset = dataset.map(transforms)

if __name__ == "__main__":
    main()
```



### Force full page OCR

This example shows how to force full-page OCR using different OCR engines like Tesseract or RapidOCR.

```python
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Any of the OCR options can be used:EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions(Mac only), RapidOcrOptions
    # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
    # ocr_options = OcrMacOptions(force_full_page_ocr=True)
    # ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(input_doc_path).document
    md = doc.export_to_markdown()
    print(md)

if __name__ == "__main__":
    main()
```



### Automatic OCR language detection with tesseract

This example demonstrates how to automatically detect the language for OCR using Tesseract.

```python
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    # Set lang=["auto"] with a tesseract OCR engine: TesseractOcrOptions, TesseractCliOcrOptions
    # ocr_options = TesseractOcrOptions(lang=["auto"])
    ocr_options = TesseractCliOcrOptions(lang=["auto"])

    pipeline_options = PdfPipelineOptions(
        do_ocr=True, force_full_page_ocr=True, ocr_options=ocr_options
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(input_doc_path).document
    md = doc.export_to_markdown()
    print(md)

if __name__ == "__main__":
    main()
```



### RapidOCR with custom OCR models

This example shows how to use RapidOCR with custom OCR models for document conversion.

```python
import os

from huggingface_hub import snapshot_download

from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
)

def main():
    # Source document to convert
    source = "https://arxiv.org/pdf/2408.09869v4"

    # Download RappidOCR models from HuggingFace
    print("Downloading RapidOCR models")
    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

    # Setup RapidOcrOptions for english detection
    det_model_path = os.path.join(
        download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx"
    )
    ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )

    pipeline_options = PdfPipelineOptions(
        ocr_options=ocr_options,
    )

    # Convert the document
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        },
    )

    conversion_result: ConversionResult = converter.convert(source=source)
    doc = conversion_result.document
    md = doc.export_to_markdown()
    print(md)

if __name__ == "__main__":
    main()
```



### Accelerator options

This example demonstrates how to configure accelerator options for Docling, allowing you to specify the number of threads and the device (CPU, MPS, CUDA) for processing.

```python
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

def main():
    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    # Explicitly set the accelerator
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.AUTO
    # )
    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CPU
    )
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.MPS
    # )
    # accelerator_options = AcceleratorOptions(
    #     num_threads=8, device=AcceleratorDevice.CUDA
    # )

    # easyocr doesnt support cuda:N allocation, defaults to cuda:0
    # accelerator_options = AcceleratorOptions(num_threads=8, device="cuda:1")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = True

    # Convert the document
    conversion_result = converter.convert(input_doc_path)
    doc = conversion_result.document

    # List with total time per document
    doc_conversion_secs = conversion_result.timings["pipeline_total"].times

    md = doc.export_to_markdown()
    print(md)
    print(f"Conversion secs: {doc_conversion_secs}")

if __name__ == "__main__":
    main()
```



### Simple translation

This example demonstrates how to perform simple translation of document content using Docling.

```python
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_translation = True
    pipeline_options.translation_options.target_lang = "es"

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    end_time = time.time() - start_time

    _log.info(f"Document converted and translated in {end_time:.2f} seconds.")

    md = conv_res.document.export_to_markdown()
    print(md)

if __name__ == "__main__":
    main()
```



### Conversion of CSV files

This example demonstrates how to convert CSV files into Docling's document representation.

```python
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import CsvFormatOption, DocumentConverter

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "csv/sample.csv"

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.CSV: CsvFormatOption(delimiter=",", header=True)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

    md = conv_res.document.export_to_markdown()
    print(md)

if __name__ == "__main__":
    main()
```



### Conversion of custom XML

This example demonstrates how to convert custom XML files into Docling's unified document representation, `DoclingDocument`, and leverage its rich structured content for RAG applications.

```python
import json
from io import BytesIO
from pathlib import Path
import tarfile
import requests
import os
from warnings import filterwarnings
from tempfile import mkdtemp

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from docling.exceptions import ConversionError

# Setup for RAG with AI dev frameworks (from the original example)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
EMBED_MODEL = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)
TEMP_DIR = Path(mkdtemp())
MILVUS_URI = str(TEMP_DIR / "docling.db")
GEN_MODEL = HuggingFaceInferenceAPI(
    token=os.getenv("HF_TOKEN"), # Assuming HF_TOKEN is set in environment
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Simple conversion of a sample PMC article
source_pmc = "../../tests/data/jats/elife-56337.nxml" # Placeholder path, replace with actual if running locally
converter = DocumentConverter()

try:
    result_pmc = converter.convert(source_pmc)
    print(f"PMC Conversion Status: {result_pmc.status}")
    md_doc_pmc = result_pmc.document.export_to_markdown()
    print("\n--- PMC Article Markdown (first 8 lines) ---")
    print("\n".join(md_doc_pmc.split("\n")[:8]))
except ConversionError as ce:
    print(f"PMC Conversion Error: {ce}")

# Example of handling unsupported XML
xml_content = (
    b'<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE docling_test SYSTEM '
    b'"test.dtd"><docling>Random content</docling>'
)
stream = DocumentStream(name="docling_test.xml", stream=BytesIO(xml_content))
try:
    result_unsupported = converter.convert(stream)
except ConversionError as ce:
    print(f"\nUnsupported XML Conversion Error: {ce}")

# Fetching data (simplified for example, actual download logic from original example)
# This part would involve downloading and extracting XML files from FTP/BDSS
# For demonstration, we'll just show the setup for the RAG part.

# Example of setting up Docling reader and node parser for LlamaIndex
# from llama_index.readers.docling import DoclingReader
# from llama_index.node_parser.docling import DoclingNodeParser
# from llama_index.vector_stores.milvus import MilvusVectorStore
# from llama_index.indices.vector_store import VectorStoreIndex
# from llama_index.query_engine import RetrieverQueryEngine

# reader = DoclingReader()
# node_parser = DoclingNodeParser()
# vector_store = MilvusVectorStore(uri=MILVUS_URI, dim=embed_dim)

# # Ingest documents (simplified, actual ingestion would involve processing fetched XMLs)
# # documents = reader.load_data(file=Path("path/to/your/xml/file.xml"))
# # nodes = node_parser.get_nodes_from_documents(documents)
# # vector_store.add(nodes)

# # Query engine setup (simplified)
# # index = VectorStoreIndex.from_vector_store(vector_store, embed_model=EMBED_MODEL)
# # query_engine = index.as_query_engine(llm=GEN_MODEL)

# # Example query
# # response = query_engine.query("What is the main topic of the document?")
# # print(f"\nQuery Response: {response}")

print("\nFurther steps for RAG with custom XML involve setting up LlamaIndex components (reader, node parser, vector store) and performing question-answering.")
```



## ✂️ Serialization & chunking

### Serialization

This section covers how Docling documents are serialized, allowing them to be saved and loaded for later use or integration with other systems.

```python
import json
import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Export Deep Search document JSON format:
    with (output_dir / f"{doc_filename}.json").open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(conv_res.document.export_to_dict()))

    # Export Text format:
    with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_res.document.export_to_text())

    # Export Markdown format:
    with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_res.document.export_to_markdown())

    # Export Document Tags format:
    with (output_dir / f"{doc_filename}.doctags").open("w", encoding="utf-8") as fp:
        fp.write(conv_res.document.export_to_document_tokens())

    end_time = time.time() - start_time

    _log.info(f"Document converted in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
```



### Hybrid chunking

This example demonstrates how to use hybrid chunking, which combines different chunking strategies to optimize document processing for various use cases.

```python
import logging
import time
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Initialize the HybridChunker
    chunker = HybridChunker()

    # Get chunks from the document
    chunks = chunker.get_chunks(conv_res.document)

    # Export chunks to a Markdown file
    with (output_dir / f"{doc_filename}_chunks.md").open("w", encoding="utf-8") as fp:
        for i, chunk in enumerate(chunks):
            fp.write(f"## Chunk {i+1}\n")
            fp.write(chunk.text)
            fp.write("\n\n")

    end_time = time.time() - start_time

    _log.info(f"Document converted and chunked in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
```



### Advanced chunking & serialization

This example delves into advanced chunking and serialization techniques, providing fine-grained control over how documents are processed and stored.

```python
import logging
import time
from pathlib import Path

from docling.chunking import DoclingChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/2206.01062.pdf"
    output_dir = Path("scratch")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()

    conv_res = doc_converter.convert(input_doc_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Initialize the DoclingChunker with custom parameters
    chunker = DoclingChunker(
        chunk_size=512,  # Max number of tokens per chunk
        chunk_overlap=50,  # Overlap between chunks
        min_chunk_size=10,  # Minimum number of tokens per chunk
        combine_adjacent_text_chunks=True,  # Combine adjacent text chunks
        split_by_page=False,  # Do not split chunks by page boundaries
    )

    # Get chunks from the document
    chunks = chunker.get_chunks(conv_res.document)

    # Export chunks to a JSON file, including metadata
    chunks_data = [
        {
            "text": chunk.text,
            "metadata": chunk.meta.model_dump_json(),
            "embedding": chunk.embedding, # If embeddings were generated
        }
        for chunk in chunks
    ]
    with (output_dir / f"{doc_filename}_advanced_chunks.json").open("w", encoding="utf-8") as fp:
        json.dump(chunks_data, fp, indent=4)

    end_time = time.time() - start_time

    _log.info(f"Document converted, chunked, and serialized in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()
```



## 🤖 RAG with AI dev frameworks

### RAG with Haystack

This example leverages the Haystack Docling extension, along with Milvus-based document store and retriever instances, as well as sentence-transformers embeddings.

```python
import os
from pathlib import Path
from tempfile import mkdtemp

from docling_haystack.converter import ExportType
from dotenv import load_dotenv

from docling.chunking import HybridChunker

# Helper function to get environment variables (from original example)
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata
        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
PATHS = ["https://arxiv.org/pdf/2408.09869"]  # Docling Technical Report
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
EXPORT_TYPE = ExportType.DOC_CHUNKS
QUESTION = "Which are the main AI models in Docling?"
TOP_K = 3
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")

# Indexing pipeline
from haystack import Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from milvus_haystack import MilvusDocumentStore, MilvusEmbeddingRetriever

document_store = MilvusDocumentStore(
    connection_args={"uri": MILVUS_URI},
    drop_old=True,
    text_field="txt",  # set for preventing conflict with same-name metadata field
)

idx_pipe = Pipeline()
idx_pipe.add_component(
    "converter",
    DoclingConverter(
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    ),
)
idx_pipe.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model=EMBED_MODEL_ID),
)
idx_pipe.add_component("writer", DocumentWriter(document_store=document_store))
if EXPORT_TYPE == ExportType.DOC_CHUNKS:
    idx_pipe.connect("converter", "embedder")
elif EXPORT_TYPE == ExportType.MARKDOWN:
    idx_pipe.add_component(
        "splitter",
        DocumentSplitter(split_by="sentence", split_length=1),
    )
    idx_pipe.connect("converter.documents", "splitter.documents")
    idx_pipe.connect("splitter.documents", "embedder.documents")
else:
    raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
idx_pipe.connect("embedder", "writer")

# To run the indexing pipeline, uncomment the following line:
# idx_pipe.run({"converter": {"paths": PATHS}})

# RAG pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils import Secret

prompt_template = """
    Given these documents, answer the question.
    Documents:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}
    Question: {{query}}
    Answer:
    """

rag_pipe = Pipeline()
rag_pipe.add_component(
    "embedder",
    SentenceTransformersTextEmbedder(model=EMBED_MODEL_ID),
)
rag_pipe.add_component(
    "retriever",
    MilvusEmbeddingRetriever(document_store=document_store, top_k=TOP_K),
)
rag_pipe.add_component("prompt_builder", PromptBuilder(template=prompt_template))
rag_pipe.add_component(
    "llm",
    HuggingFaceAPIGenerator(
        api_type="serverless_inference_api",
        api_params={"model": GENERATION_MODEL_ID},
        token=Secret.from_token(HF_TOKEN) if HF_TOKEN else None,
    ),
)
rag_pipe.add_component("answer_builder", AnswerBuilder())
rag_pipe.connect("embedder.embedding", "retriever")
rag_pipe.connect("retriever", "prompt_builder.documents")
rag_pipe.connect("prompt_builder", "llm")
rag_pipe.connect("llm.replies", "answer_builder.replies")
rag_pipe.connect("llm.meta", "answer_builder.meta")
rag_pipe.connect("retriever", "answer_builder.documents")

# To run the RAG pipeline, uncomment the following lines:
# rag_res = rag_pipe.run(
#     {
#         "embedder": {"text": QUESTION},
#         "prompt_builder": {"query": QUESTION},
#         "answer_builder": {"query": QUESTION},
#     }
# )

# from docling.chunking import DocChunk
# print(f"Question:\n{QUESTION}\n")
# print(f"Answer:\n{rag_res['answer_builder']['answers'][0].data.strip()}\n")
# print("Sources:")
# sources = rag_res["answer_builder"]["answers"][0].documents
# for source in sources:
#     if EXPORT_TYPE == ExportType.DOC_CHUNKS:
#         doc_chunk = DocChunk.model_validate(source.meta["dl_meta"])
#         print(f"- text: {doc_chunk.text!r}")
#         if doc_chunk.meta.origin:
#             print(f"  file: {doc_chunk.meta.origin.filename}")
#         if doc_chunk.meta.headings:
#             print(f"  section: {' / '.join(doc_chunk.meta.headings)}")
#         bbox = doc_chunk.meta.doc_items[0].prov[0].bbox
#         print(
#             f"  page: {doc_chunk.meta.doc_items[0].prov[0].page_no}, "
#             f"bounding box: [{int(bbox.l)}, {int(bbox.t)}, {int(bbox.r)}, {int(bbox.b)}]"
#         )
#     elif EXPORT_TYPE == ExportType.MARKDOWN:
#         print(repr(source.content))
#     else:
#         raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
```



### RAG with LangChain

This example demonstrates how to integrate Docling with LangChain for RAG applications, enabling document processing and retrieval within the LangChain framework.

```python
import logging
import os
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from langchain_community.document_loaders import DoclingLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Helper function to get environment variables (from original example)
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata
        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
OPENAI_API_KEY = _get_env_from_colab_or_os("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)

# Define the path to the document to be processed
DATA_PATH = Path("../../tests/data/pdf/2206.01062.pdf")

# 1. Load documents using DoclingLoader
# This will convert the document into a format suitable for LangChain
loader = DoclingLoader(file_path=DATA_PATH)
docs = loader.load()

# 2. Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a vector store (using Milvus for this example)
# Ensure Milvus is running or configure it to use a local file
MILVUS_URI = str(Path(mkdtemp()) / "docling_langchain.db")
vectorstore = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="docling_langchain_collection",
)

# 4. Define the retriever
retriever = vectorstore.as_retriever()

# 5. Define the LLM (using OpenAI for this example)
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 6. Create a prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. Build the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# 8. Invoke the chain with a question
question = "What is the main topic of the document?"
# response = chain.invoke(question)
# print(f"\nQuestion: {question}")
# print(f"Answer: {response}")

print("LangChain RAG setup complete. Uncomment the chain.invoke() line to run a query.")
```



### RAG with LlamaIndex

This example demonstrates how to integrate Docling with LlamaIndex for RAG applications, enabling document processing and retrieval within the LlamaIndex framework.

```python
import logging
import os
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.indices.vector_store import VectorStoreIndex

# Helper function to get environment variables (from original example)
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata
        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")

logging.basicConfig(level=logging.INFO)

# Define paths and models
DATA_PATH = Path("../../tests/data/pdf/2206.01062.pdf")
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TEMP_DIR = Path(mkdtemp())
MILVUS_URI = str(TEMP_DIR / "docling_llamaindex.db")

# 1. Initialize DoclingReader and DoclingNodeParser
reader = DoclingReader()
node_parser = DoclingNodeParser()

# 2. Load documents using DoclingReader
documents = reader.load_data(file=DATA_PATH)

# 3. Parse documents into nodes using DoclingNodeParser
nodes = node_parser.get_nodes_from_documents(documents)

# 4. Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)

# 5. Initialize LLM
llm = HuggingFaceInferenceAPI(
    token=HF_TOKEN,
    model_name=GEN_MODEL_ID,
)

# 6. Initialize Milvus vector store
# Get embedding dimension from a sample embedding
embed_dim = len(embed_model.get_text_embedding("test"))
vector_store = MilvusVectorStore(uri=MILVUS_URI, dim=embed_dim)

# 7. Create a vector store index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model,
    nodes=nodes, # Pass nodes directly to the index for ingestion
)

# 8. Create a query engine
query_engine = index.as_query_engine(llm=llm)

# 9. Query the engine
question = "What is the abstract of the document?"
# response = query_engine.query(question)
# print(f"\nQuestion: {question}")
# print(f"Answer: {response}")

print("LlamaIndex RAG setup complete. Uncomment the query_engine.query() line to run a query.")
```



### RAG with LangChain

This example demonstrates how to integrate Docling with LangChain for RAG applications, enabling document processing and retrieval within the LangChain framework.

```python
import logging
import os
from pathlib import Path
from tempfile import mkdtemp

from dotenv import load_dotenv
from langchain_community.document_loaders import DoclingLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Helper function to get environment variables (from original example)
def _get_env_from_colab_or_os(key):
    try:
        from google.colab import userdata
        try:
            return userdata.get(key)
        except userdata.SecretNotFoundError:
            pass
    except ImportError:
        pass
    return os.getenv(key)

load_dotenv()
HF_TOKEN = _get_env_from_colab_or_os("HF_TOKEN")
OPENAI_API_KEY = _get_env_from_colab_or_os("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)

# Define the path to the document to be processed
DATA_PATH = Path("../../tests/data/pdf/2206.01062.pdf")

# 1. Load documents using DoclingLoader
# This will convert the document into a format suitable for LangChain
loader = DoclingLoader(file_path=DATA_PATH)
docs = loader.load()

# 2. Define the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create a vector store (using Milvus for this example)
# Ensure Milvus is running or configure it to use a local file
MILVUS_URI = str(Path(mkdtemp()) / "docling_langchain.db")
vectorstore = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"uri": MILVUS_URI},
    collection_name="docling_langchain_collection",
)

# 4. Define the retriever
retriever = vectorstore.as_retriever()

# 5. Define the LLM (using OpenAI for this example)
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# 6. Create a prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. Build the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

# 8. Invoke the chain with a question
question = "What is the main topic of the document?"
# response = chain.invoke(question)
# print(f"\nQuestion: {question}")
# print(f"Answer: {response}")

print("LangChain RAG setup complete. Uncomment the chain.invoke() line to run a query.")
```



