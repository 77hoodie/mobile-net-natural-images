# Classificação de imagens naturais com MobileNetV2

Projeto da disciplina de Inteligência Computacional / Machine Learning (2026.2), voltado à classificação multiclasse de imagens por transfer learning e fine-tuning com a arquitetura MobileNetV2.

**Etapa atual:** defesa intermediária — baseline treinada e avaliada na validação; estratégias de fine-tuning planejadas.

## Dataset

O [Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images) contém 6.899 imagens de oito classes: avião, carro, gato, cachorro, flor, fruta, motocicleta e pessoa. A variável-alvo corresponde à categoria da imagem.

| Conjunto | Imagens | Proporção aproximada |
|---|---:|---:|
| Treino | 4.829 | 70% |
| Validação | 1.035 | 15% |
| Teste | 1.035 | 15% |

A divisão estratificada utiliza semente 42 e é preservada em `splits/holdout.csv`. Foi verificada a ausência de duplicatas exatas compartilhadas entre os subconjuntos. O teste permanece reservado para a etapa final. O dataset não é versionado no Git: a obtenção é realizada pelo KaggleHub.

## Metodologia

- MobileNetV2 com pesos ImageNet, `alpha=1.0` e base convolucional congelada.
- Entrada RGB de 224 × 224 pixels, com preservação da proporção e normalização para [-1,1].
- Aumentação exclusiva no treino: espelhamento horizontal e rotação de até ±18°.
- Global Average Pooling, Dropout de 0,2 e camada Dense com oito saídas softmax.
- Adam (learning rate 0,001), batch size 32, até 20 épocas e early stopping com paciência 4.
- Seleção do checkpoint pelo maior F1 macro de validação.

Dos 2.268.232 parâmetros do modelo, 10.248 são treináveis na condição de referência.

## Resultados parciais

| Indicador | Resultado na validação |
|---|---:|
| Acurácia | 99,81% |
| F1 macro | 99,78% |
| Balanced accuracy | 99,77% |
| Loss | 0,008307 |
| Acertos | 1.033 / 1.035 |
| Melhor época | 5 |
| Épocas executadas | 9 |
| Tempo de treinamento | 17 min 33 s (CPU) |

Os dois erros de classificação ocorreram na classe avião, confundida uma vez com cachorro e uma vez com flor. O desempenho indica a utilidade das representações pré-treinadas para o dataset, mas não demonstra generalização para imagens externas. A análise das curvas, da matriz de confusão e dos exemplos visuais está documentada no notebook.

## Estratégias de fine-tuning

| Estratégia | Camadas ajustáveis da base | Parâmetros treináveis | Learning rate |
|---|---|---:|---:|
| A | Blocos 13–16 e `Conv_1` | 1.673.608 | 0,00001 |
| B | Blocos 6–16 e `Conv_1` | 2.148.040 | 0,00001 |

Ambas iniciarão a partir de cópias independentes do checkpoint da baseline, com Batch Normalization congelada. Nesta etapa, foram apenas configuradas e inspecionadas; ainda não há resultados de treinamento A/B, Grad-CAM ou teste.

## Execução e arquivos

O notebook foi executado com TensorFlow 2.20.0 em CPU. As dependências são informadas na primeira célula de código. A execução requer acesso ao Kaggle e aos pesos ImageNet quando não disponíveis em cache. Para uso de uma cópia local do dataset, pode-se definir a variável de ambiente `NATURAL_IMAGES_DIR`.

O [notebook executado](mobilenetv2_natural_images.ipynb) contém as saídas e a análise dos resultados. A visualização não exige novo treinamento. A execução integral das células inicia um novo experimento e substitui o checkpoint e os registros da baseline; recomenda-se preservar os arquivos atuais caso seja necessário reproduzir o treinamento.

- `mobilenetv2_natural_images.ipynb`: metodologia, código, figuras e análise crítica.
- `splits/holdout.csv`: manifesto da divisão do dataset.
- `artifacts/baseline_best.keras`: checkpoint selecionado na validação.
- `artifacts/baseline_config.json`: configuração do experimento.
- `artifacts/baseline_history.csv`: histórico de treinamento.
- `artifacts/baseline_val_metrics.csv`, `baseline_val_report.csv` e `baseline_val_predictions.csv`: resultados de validação.
- `artifacts/visual_cases_val.csv`: imagens selecionadas para análise comparativa futura.

## Referências

- Sandler et al. (2018). [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html). CVPR.
- [Natural Images — Kaggle](https://www.kaggle.com/datasets/prasunroy/natural-images).
- Keras. [MobileNet models](https://keras.io/api/applications/mobilenet/mobilenet_models/) e [Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

A declaração de uso de IA generativa está registrada na seção inicial do notebook.
