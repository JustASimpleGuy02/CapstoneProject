# FASHION RECOMMENDER SYSTEM BY PROMPS

## Setup
```
$ conda create -n fclip python=3.11
$ conda activate fclip
$ pip install -r requirements.txt
$ pip install dill==0.3.7 # to speedup
```

## Try your Prompt
```
$ streamlit run outfit_recommend_app.py
```

## Reference

@Article{Chia2022,
    title="Contrastive language and vision learning of general fashion concepts",
    author="Chia, Patrick John
            and Attanasio, Giuseppe
            and Bianchi, Federico
            and Terragni, Silvia
            and Magalh{\~a}es, Ana Rita
            and Goncalves, Diogo
            and Greco, Ciro
            and Tagliabue, Jacopo",
    journal="Scientific Reports",
    year="2022",
    month="Nov",
    day="08",
    volume="12",
    number="1",
    pages="18958",
    abstract="The steady rise of online shopping goes hand in hand with the development of increasingly complex ML and NLP models. While most use cases are cast as specialized supervised learning problems, we argue that practitioners would greatly benefit from general and transferable representations of products. In this work, we build on recent developments in contrastive learning to train FashionCLIP, a CLIP-like model adapted for the fashion industry. We demonstrate the effectiveness of the representations learned by FashionCLIP with extensive tests across a variety of tasks, datasets and generalization probes. We argue that adaptations of large pre-trained models such as CLIP offer new perspectives in terms of scalability and sustainability for certain types of players in the industry. Finally, we detail the costs and environmental impact of training, and release the model weights and code as open source contribution to the community.",
    issn="2045-2322",
    doi="10.1038/s41598-022-23052-9",
    url="https://doi.org/10.1038/s41598-022-23052-9"
}
