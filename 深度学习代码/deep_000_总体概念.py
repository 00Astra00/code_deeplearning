"""
人工智能>机器学习>深度学习

机器学习算法（需要结构化数据）包括监督学习、无监督学习、强化学习

1.监督学习：
数据已经被标记，这意味着你知道目标变量
系统可以根据过去的数据预测未来的结果
要求至少为模型提供一个输入和输出变量
常见技术：线性回归，逻辑回归，支持向量机、朴素贝叶斯和逻辑树

2.无监督学习：
数据未被标记
自行从数据中发现模式，也就是说系统能够从提供的输入数据中识别隐藏的特征，一旦数据更具可读性、模式和相似性——更明显
常见技术：k、均值聚类、层次聚类和异常检测

3.强化学习（再励学习）：
描述和解决智能体在和环境的交互过程中，通过学习策略以达成回报最大化或实现特定目标的问题。
7个步骤：收集数据、数据预处理、选择模型、训练模型、评估模型、调整参数以及预测。


深度学习： 复杂算法和深层（需要多层）神经网络在机器学习中的应用

受人脑结构和功能启发的算法
可以处理大量结构化和非结构化数据
(chatgpt属于深度学习中的自然语言处理)
工作方式：首先计算输入的加权和，然后计算的权重总和作为输入传递给激活函数，再加入一个偏差，然后决定神经元是否应该被激发，接着输出层给出预测的输出。
输出模型还会与实际输出进行比较，训练好神经网络后，模型还会用反向传播的方法实现更高的精度
"""

"""
多模态核心总结
多模态学习（Multimodal Learning） 的核心思想是模仿人类综合利用多种感官（视觉、听觉、触觉等）来认知世界的方式，
让机器能够协同处理和理解来自不同源头（模态）的数据（如文本、图像、音频、视频），从而实现比单一模态更强大、更鲁棒的感知与理解能力。

其技术框架可归纳为三个核心环节：

一、模态表示 (Modal Representation) - “如何理解单一模态”
这是多模态处理的基础步骤，旨在将各种原始数据转换为计算机可处理的数值化向量（嵌入，Embedding），并保留其关键信息。

文本模态：主流方法是使用预训练语言模型（如BERT）获得上下文相关的词/句向量。

视觉模态：

图像：通常使用卷积神经网络（CNN）（如ResNet）提取空间特征。

视频：需同时处理空间和时间信息，常用CNN + RNN/LSTM 或 3D-CNN 等模型。

声音模态：通过信号处理和深度学习模型（如CNN, RNN）提取声学特征。

两种高级表示策略：

联合表示 (Joint Representation)：将不同模态的信息早期融合，映射到一个统一的向量空间中。适用于模态紧密相关的任务。

协同表示 (Coordinated Representation)：将不同模态的信息分别映射到各自的向量空间，但通过约束（如相似度）让这些空间结构对齐。适用于模态差异较大或需要灵活检索的任务。

二、多模态融合 (Multimodal Fusion) - “如何综合多种信息”
融合是将不同模态的信息整合起来，发挥“1+1>2”效果的关键。根据融合发生的阶段，可分为三个层次：

数据级融合：在最底层直接合并原始数据（如图像像素与深度图叠加）。简单但要求数据高度对齐且冗余度高。

特征级融合：在中间层将各模态提取出的特征向量进行拼接、加权或通过神经网络融合。这是最常用的策略，能有效保留各模态特性并实现互补。

决策级融合：在最顶层让不同模态的模型先独立做出决策（如分类结果），再对多个决策结果进行综合（如投票、加权平均）。灵活性高，容错性好，但可能忽略模态间的细微交互。

三、跨模态对齐 (Cross-Modal Alignment) - “如何建立模态间的联系”
对齐是让模型理解“什么文本对应什么图像”或“什么声音对应什么动作”的能力，是实现跨模态检索、生成等任务的核心。分为显式和隐式两种：

显式对齐 (Explicit Alignment)：直接地、有监督地建立模态元素间的对应关系。

监督对齐：利用成对的标注数据（如图文对），通过损失函数直接拉近相关样本在向量空间中的距离。

无监督对齐：利用数据的统计特性（如CCA）或重构约束（如循环一致性损失）来发现对应关系，无需配对标签。

隐式对齐 (Implicit Alignment)：间接地通过模型内部机制学会对齐，而无需显式的对应标签。

注意力对齐：模型（如Transformer）通过注意力机制自动学习并关注不同模态间最相关的部分。

语义对齐：通过对比学习（如CLIP）或在大规模数据上预训练，让模型在高级语义层面将不同模态的同类概念映射到向量空间的相近位置。

核心思想与价值
多模态学习的终极目标是让AI获得融会贯通的能力。表示是基础，解决了“数据如何入脑”的问题；融合是手段，解决了“如何综合思考”的问题；对齐是灵魂，解决了“如何举一反三”的问题。
这三者相辅相成，共同构成了多模态AI强大能力的基石，推动了如图像描述、视觉问答、跨模态检索等一系列前沿应用的发展。
"""

"""
多模态情感分析核心内容总结
本文系统地介绍了多模态情感分析（Multimodal Sentiment Analysis）这一研究领域，重点围绕其任务定义、价值、相关数据集和核心方法进行了梳理。

一、核心概念与价值
定义：多模态情感分析旨在综合利用文本、图像、音频（及视频） 等多种模态的数据，来识别和理解其中所蕴含的情感。

价值与挑战：

价值：多模态数据包含更丰富的信息，模态间可以互补（如识别文本反讽需要图像上下文），使得机器能更自然、更准确地理解人类情感，是人机交互发展的重要方向。

挑战：关键在于如何有效融合不同模态的信息，并建模它们之间的关联与对齐，以避免引入噪声，真正发挥“1+1>2”的效果。

二、主要任务分类
文章按模态组合将任务分为两大类：

面向图文（文本+图像）的任务：

图文情感分类：判断整条图文内容（如商品评论、社交媒体推文）的情感倾向。

图文方面级情感分类：针对文本中提到的特定方面（如手机的“电池”、“屏幕”），分析其对应的情感。

图文反讽识别：判断图文内容是否在表达反讽（如文字说“真好”但配图内容相反）。

面向视频（文本+图像+音频）的任务：

评论视频情感分类：判断视频中讲述者（如产品评测博主）的情感倾向。

对话视频情感分类：在多人对话场景中，判断每位说话者在特定回合的情感，需考虑对话上下文和说话人信息。

新闻视频情感分类：分析新闻节目中人物的情感。

对话视频反讽识别：判断视频对话中是否包含反讽表达。

三、代表性数据集与方法
文章详细总结了13个公开数据集（5个图文，8个视频）并介绍了各类任务的典型方法。

图文情感分类：

数据集：Yelp（餐厅评论）、Tumblr（多情绪推文）。

方法：VistaNet - 使用视觉注意力机制，让图像来指导文本建模，确定文本中哪些句子对情感判断更重要。

图文方面级情感分类：

数据集：Multi-ZOL（手机方面评论）、Twitter-15/17（推文实体情感）。

方法：MIMN - 使用方面引导的注意力（Aspect-guided Attention） 和多交互注意力（Multi-interactive Attention） 机制来融合文本和图像信息，并捕捉模态内和模态间的交互。

图文反讽识别：

数据集：Twitter Sarcasm（带反讽标签的推文）。

方法：HFM（分层融合模型） - 引入图像属性（如图片中的物体标签）作为连接文本和图像的“桥梁”，通过表示融合层和模态融合层进行分层信息融合。

评论视频情感分类：

数据集：YouTube, ICT-MMMO, MOSI, CMU-MOSEI（均为单人评论视频）。

方法：

TFN（张量融合网络）：利用张量外积进行融合，同时捕获单模态、双模态和三模态的交互信息。

MARN（多注意力循环网络）：使用多级注意力机制来捕捉多种不同的模态间交互模式。

MFN（记忆融合网络）：通过Delta记忆注意力和多视角门控记忆来捕捉时序上和模态间的交互。

对话视频情感分类：

数据集：MELD（来自《老友记》），IEMOCAP（演员表演）。

方法：DialogueRNN - 使用多个GRU分别对说话人信息、全局对话上下文和情感信息进行建模，专门处理对话的动态特性。

四、结论
多模态情感分析因实际需求而迅速发展，其核心研究问题始终是：如何高效地处理、融合不同模态的信息，并利用它们之间的互补性来提升情感理解的性能。
尽管已有众多数据集和创新模型（如各种注意力、记忆、融合机制），该领域在如何最优地实现多模态融合与对齐方面，仍然面临持续的研究挑战。
"""

"""
# 激活 conda 环境
source /opt/conda/bin/activate
# 在 Filebrowser 上可以上传整个文件夹


import torch
import clip
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import json
import os
from tqdm import tqdm

class MultimodalFeatureExtractor:
    def __init__(self):
        # 加载CLIP模型（文本-图像多模态模型）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 加载ViT模型
        self.vit_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(self.device)
    
    def extract_image_features(self, image_path):
        """提取图像视觉特征"""
        try:
            if os.path.exists(image_path):
                # 方法1: 使用CLIP提取特征
                image = Image.open(image_path).convert('RGB')
                clip_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    clip_features = self.clip_model.encode_image(clip_image)
                
                # 方法2: 使用ViT提取详细特征
                vit_inputs = self.vit_extractor(image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    vit_outputs = self.vit_model(**vit_inputs)
                    vit_features = vit_outputs.last_hidden_state.mean(dim=1)
                
                return {
                    "clip_features": clip_features.cpu().numpy().tolist(),
                    "vit_features": vit_features.cpu().numpy().tolist(),
                    "image_caption": self.generate_caption(clip_features)
                }
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        
        return {
            "clip_features": [[0]*512],
            "vit_features": [[0]*768], 
            "image_caption": ""
        }
    
    def generate_caption(self, image_features):
        """生成图像描述（简化版）"""
        # 这里可以集成图像描述模型如BLIP
        return "image with visual content related to text"
    
    def extract_face_emotion(self, image_path):
        """提取面部情感特征"""
        try:
            # 这里可以集成面部检测和情感分析模型
            return {
                "face_detected": True,
                "emotion": "neutral",
                "face_features": [[0]*256]
            }
        except:
            return {
                "face_detected": False,
                "emotion": "neutral", 
                "face_features": [[0]*256]
            }

def enhance_with_multimodal_features():
    """为现有数据添加多模态特征"""
    extractor = MultimodalFeatureExtractor()
    
    datasets = ['twitter2015', 'twitter2017']
    
    for dataset in datasets:
        for split in ['train', 'val', 'test']:
            input_file = f'./dataset/{dataset}_corenlp/{dataset}_{split}_write.json'
            output_file = f'./dataset/{dataset}_multimodal/{dataset}_{split}_multimodal.json'
            
            if not os.path.exists(input_file):
                continue
                
            print(f"Processing {input_file}...")
            
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            enhanced_data = []
            
            for item in tqdm(data):
                # 提取图像特征
                image_id = item.get('image_id', '')
                image_path_2015 = f'./raw_data/twitter2015_images/{image_id}.jpg'
                image_path_2017 = f'./raw_data/twitter2017_images/{image_id}.jpg'
                
                image_path = image_path_2015 if os.path.exists(image_path_2015) else image_path_2017
                
                # 多模态特征提取
                image_features = extractor.extract_image_features(image_path)
                face_emotion = extractor.extract_face_emotion(image_path)
                
                # 增强数据
                enhanced_item = item.copy()
                enhanced_item.update({
                    "multimodal_features": {
                        "visual": image_features,
                        "face_emotion": face_emotion,
                        "fusion_ready": True
                    }
                })
                
                enhanced_data.append(enhanced_item)
            
            # 保存增强数据
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(enhanced_data, f, indent=2)
            
            print(f"Enhanced {len(enhanced_data)} samples with multimodal features")

if __name__ == '__main__':
    enhance_with_multimodal_features()
"""