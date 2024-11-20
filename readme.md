

## To be able to use openAI API, users must create a simple `.env` file with your API key:

```
OPENAI_API_KEY="skxxxxxxxxxxxxx"
```



#### PS: The complte organization of this project is something like this:

```
├── cache/ # Directory for cached files for GPT-genrated text
├── configs/ # Configuration files 
├── data/ # Data files and datasets 
├── figures/ # Generated figures and visualizations 
├── models/ # duprecated code
├── pseudo_images/ # SDXL generated images, not so useful here
├── results/ # Results and outputs 
├── utils/ # Utility scripts and functions 
├── .env # Environment variables 
├── *.ipynb # jupyter notebooks that contains many on-going experiments, very messy
└── .gitignore # Git ignore file
```

I put many of the directories into .gitignore file to avoid taking too much space