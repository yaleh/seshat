import gradio as gr
import numpy as np
import pandas as pd
import tempfile
import uuid
import ast
from pinecone import Pinecone as PineconeClient
from sklearn.cluster import SpectralClustering
from pymilvus import MilvusClient
from components.lcel import EmbeddingFactory
from tools.table_parser import TableParser
from tools.utils import detect_encoding

# Define an array of model settings
model_settings = [
    {
        "model": "text-embedding-ada-002",
    },
    {
        "model": "text-embedding-3-small",
    },
    {
        "model": "text-embedding-3-large",
    },
]

class EmbeddingUI:
    """
    A class representing the user interface for embedding data using OpenAI models and Pinecone.

    Attributes:
        ui (gr.Blocks): The user interface layout.

    Methods:
        __init__(): Initializes the EmbeddingUI class.
        init_ui(): Initializes the user interface layout.
        run_agent(input_text, model_name): Runs the agent to embed the input text using the specified model.
        embed_dataframe(input_dataframe, model_name): Embeds the input dataframe using the specified model.
        upload_data(file): Uploads the data from the specified file.
        upload_embedded(file): Uploads the embedded data from the specified file.
        import_data(pinecone_host, pinecone_index, pinecone_api_key, output_dataframe): Imports the data to Pinecone index.
    """

    def __init__(self, config=None):
        self.config = config
        self.ui = self.init_ui()


    def init_ui(self):
        with gr.Blocks() as block:
            with gr.Row():
                with gr.Column(scale=1):
                    self.model_name = gr.Dropdown(
                        label="Model",
                        choices=[k for k,_ in self.config.embedding.items()],
                        value=list(self.config.embedding.keys())[0],
                    )
                    self.vdb_type = gr.Dropdown(
                        ["Pinecone", "Milvus"],
                        label="Vector Database",
                        value="Pinecone",
                        visible=False,
                        interactive=True
                    )

                    self.pinecone_tab = gr.Tab("Pinecone")
                    self.milvus_tab = gr.Tab("Milvus")

                    with self.pinecone_tab:
                        self.pinecone_host = gr.Textbox(
                            label="Pinecone Host", placeholder="Pinecone Host"
                        )
                        self.pinecone_api_key = gr.Textbox(
                            label="Pinecone API Key", placeholder="Pinecone API Key"
                        )

                    with self.milvus_tab:
                        self.milvus_uri = gr.Textbox(
                            label="Milvus URI", placeholder="Milvus URI"
                        )
                        self.milvus_token = gr.Textbox(
                            label="Milvus Token", placeholder="Milvus Token"
                        )
                        self.milvus_collection_name = gr.Textbox(
                            label="Milvus Collection Name", placeholder="Milvus Collection Name"
                        )

                    with gr.Group():
                        self.vdb_index_description = gr.Textbox(
                            label="VDB Index Description", 
                            placeholder="VDB Index Description",
                            interactive=False
                        )
                        with gr.Row():
                            self.refresh_vdb_btn = gr.Button(value="Refresh", variant='primary')
                            self.clear_vdb_btn = gr.Button(value="Clear VDB", variant='danger')

                    self.importing_batch_size = gr.Slider(
                        1, 100, value=20, label="Importing Batch Size",
                        info="The number of vectors to import in each batch"
                    )

                    with gr.Row():
                        self.file_table = gr.File(label="Table Data")

                        with gr.Group():
                            self.file_embeddings = gr.File(
                                label="Embeddings",
                                file_types=['csv', 'xls', 'xlsx']
                            )
                            self.import_data_btn = gr.Button(value="Import Data", variant='primary')

                with gr.Column(scale=2):
                    # Add a Gradio Dataframe and a Gradio File component
                    with gr.Tab("Table"):
                        self.input_dataframe = gr.Dataframe()
                        with gr.Row():
                            self.key_field = gr.Textbox(label="Key Field", value="id")
                            self.value_fields = gr.Textbox(
                                label="Value Fields",
                                placeholder="field1, field2, ...",
                                value="text"
                            )

                        with gr.Tab("Embedding"):
                            with gr.Row():
                                self.embed_dataframe_btn = gr.Button(value="Embed Dataframe", variant='primary')
                                self.embed_import_btn = gr.Button(value="Embed and Import Data", variant='primary')
                        with gr.Tab("Clustering"):
                            with gr.Row():
                                self.clusters = gr.Number(label="Clusters", value=10)
                                self.clustering_btn = gr.Button(value="Cluster Analysis", variant='primary')

                    with gr.Tab("Text"):
                        self.input_text = gr.Textbox(
                            label="Input Text",
                            show_copy_button=True,
                            interactive=True
                            )
                        self.output_embedding = gr.Textbox(
                            label="Embedding",
                            show_copy_button=True,
                            interactive=False
                            )
                        self.embed_text_btn = gr.Button(value="Embed", variant='primary')

                        self.vdb_search_output = gr.Textbox(
                            label="VDB Search Output", 
                            placeholder="VDB Search Output",
                            show_copy_button=True,
                            interactive=False
                        )
                        self.vdb_search_meta = gr.Textbox(
                            label="VDB Search Meta", 
                            placeholder="VDB Search Meta",
                            show_copy_button=True,
                            interactive=False
                        )
                        with gr.Row():
                            self.vdb_search_btn = gr.Button(value="Search VDB", variant='primary')
                            self.embed_search_btn = gr.Button(value="Embed and Search VDB", variant='primary')

            self.pinecone_tab.select(self.select_vdb_tab, [], [self.vdb_type])
            self.milvus_tab.select(self.select_vdb_tab, [], [self.vdb_type])

            self.refresh_vdb_btn.click(
                self.refresh_vdb,
                [self.vdb_type,
                 self.pinecone_host, self.pinecone_api_key, 
                 self.milvus_uri, self.milvus_token, self.milvus_collection_name],
                [self.vdb_index_description]
            )
            self.clear_vdb_btn.click(
                self.clear_vdb, 
                [self.vdb_type,
                 self.pinecone_host, self.pinecone_api_key,
                 self.milvus_uri, self.milvus_token, self.milvus_collection_name],
                [self.vdb_index_description]
            )
            self.file_table.upload(
                self.upload_data, 
                [self.file_table], 
                [self.input_dataframe]
            )
            self.embed_dataframe_btn.click(
                self.embed_dataframe,
                [self.input_dataframe, self.key_field, self.value_fields, self.model_name],
                [self.file_embeddings]
            )
            self.import_data_btn.click(
                self.import_data,
                [
                    self.vdb_type,
                    self.pinecone_host,
                    self.pinecone_api_key,
                    self.milvus_uri,
                    self.milvus_token,
                    self.milvus_collection_name,
                    self.input_dataframe,
                    self.file_embeddings,
                    self.importing_batch_size
                ],
                [self.vdb_index_description]
            )
            self.embed_import_btn.click(
                self.embed_import_dataframe,
                [
                    self.input_dataframe,
                    self.key_field,
                    self.value_fields,
                    self.model_name,
                    self.vdb_type,
                    self.pinecone_host,
                    self.pinecone_api_key,
                    self.milvus_uri,
                    self.milvus_token,
                    self.milvus_collection_name,
                    self.importing_batch_size
                ],
                [self.vdb_index_description]
            )
            self.clustering_btn.click(
                self.cluster_dataframe,
                [self.input_dataframe, self.file_embeddings, self.clusters],
                [self.input_dataframe, self.file_table]
            )

            self.embed_text_btn.click(
                self.run_agent, 
                [self.input_text, self.model_name], 
                [self.output_embedding]
            )

            self.vdb_search_btn.click(
                self.vdb_search,
                [
                    self.vdb_type,
                    self.pinecone_host,
                    self.pinecone_api_key,
                    self.milvus_uri,
                    self.milvus_token,
                    self.milvus_collection_name,
                    self.output_embedding
                ],
                [self.vdb_search_output, self.vdb_search_meta]
            )
            self.embed_search_btn.click(
                self.embed_search,
                [
                    self.model_name,
                    self.vdb_type,
                    self.pinecone_host,
                    self.pinecone_api_key,
                    self.milvus_uri,
                    self.milvus_token,
                    self.milvus_collection_name,
                    self.input_text
                ],
                [self.vdb_search_output, self.vdb_search_meta]
            )

        return block
    
    def select_vdb_tab(self, event: gr.SelectData):
        return str(event.value)            
    
    def run_agent(self, input_text, model_name):
        # model_index = next((index for index, setting in enumerate(model_settings) if setting['model'] == model_name), 0)
        # model = OpenAIEmbeddings(**model_settings[model_index])

        model = self._create_embedding_model(model_name)
        # TODO: remove '[]' from the begining and end of the output
        return model.embed_query(input_text)

    def refresh_vdb(self, vdb_type, 
                    pinecone_host, pinecone_api_key,
                    milvus_uri, milvus_token, milvus_collection_name):
        if vdb_type == "Pinecone":
            client = PineconeClient(pinecone_api_key)
            index = client.Index(host=pinecone_host)
            return index.describe_index_stats().to_str()
        elif vdb_type == "Milvus":
            milvus = MilvusClient(uri=milvus_uri, token=milvus_token)
            return milvus.get_collection_stats(milvus_collection_name)

        return "No VDB selected"

    def clear_vdb(self, vdb_type,
                  pinecone_host, pinecone_api_key,
                  milvus_uri, milvus_token, milvus_collection_name):
        if vdb_type == "Pinecone":
            client = PineconeClient(pinecone_api_key)
            index = client.Index(host=pinecone_host)
            index.delete(delete_all=True)
            gr.Info("VDB index cleared successfully")
            return index.describe_index_stats().to_str()
        elif vdb_type == "Milvus":
            client = MilvusClient(uri=milvus_uri, token=milvus_token)

            # get all ids
            ids = client.query(milvus_collection_name, filter="id >= 0", output_fields=["id"])
            # ids looks like "[{'id': 0}, {'id': 1}, ...]"
            ids = [int(id['id']) for id in ids]
            # delete all ids
            client.delete(milvus_collection_name, ids)

            gr.Info("VDB index cleared successfully")
            return client.get_collection_stats(milvus_collection_name)
        
        return "No VDB selected"

    def upload_data(self, file):
        try:
            if file.name.endswith('csv'):
                enco = detect_encoding(file.name)
                df = pd.read_csv(file.name, encoding=enco)
            elif file.name.endswith('xls') or file.name.endswith('xlsx'):
                df = pd.read_excel(file.name)
            else:
                df = pd.DataFrame()
        except Exception as e:
            raise gr.Error(e)
        return df
    
    def upload_embedded(self, file):
        # Assuming the file is a CSV file
        df = pd.read_csv(file)
        df.iloc[:, 0] = df.iloc[:, 0].apply(ast.literal_eval)
        return df

    def _create_embedding_model(self, model_name):
        factory = EmbeddingFactory()
        embedding_type = self.config.embedding[model_name].type
        embedding_args = self.config.embedding[model_name].dict()
        embedding_args.pop('type', None)

        model = factory.create(embedding_type, **embedding_args)
        return model

    def _embed_dataframe(self, input_dataframe, key_field, value_fields, model_name):
        model = self._create_embedding_model(model_name)

        # split the value_fields into a list and remove any leading or trailing whitespaces
        value_fields = [field.strip() for field in value_fields.split(",")]

        embedded_vectors = []

        key_column = input_dataframe[key_field]
        # value_columns = input_dataframe[value_fields]

        embedded_vectors = model.embed_documents(key_column.tolist())

        # create a new dataframe with the embedded_vectors (string) as the firt column named `Vector`
        embedded_table = pd.DataFrame({"Vector": embedded_vectors})

        # append value_fields to the right of the new dataframe
        embedded_table = pd.concat([embedded_table, input_dataframe[value_fields]], axis=1)

        return embedded_table, embedded_vectors

    def embed_dataframe(self, input_dataframe, key_field, value_fields, model_name):
        try:
            _, embedded_vectors = self._embed_dataframe(
                input_dataframe,
                key_field,
                value_fields,
                model_name)

            # save embedded vectors to a temporary npy file
            temp_filename = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
            np.save(temp_filename, embedded_vectors)

            return temp_filename
        except Exception as e:
            gr.Warning(f'Error: {e}')
            return None        
 
    def import_data(self, vdb_type, 
                    pinecone_host, pinecone_api_key,
                    milvus_uri, milvus_token, milvus_collection_name,
                    input_dataframe, embeddings, batch_size):
        metadatas = [{input_dataframe.columns[1]: t} for t in input_dataframe.iloc[:, 1].tolist()]
        if isinstance(embeddings, gr.File) or isinstance(embeddings, str):
            vectors = np.load(embeddings)
        else:
            vectors = embeddings

        # validate the number of vectors and metadatas
        if len(vectors) != len(metadatas):
            gr.Warning("The number of vectors and metadatas should be the same")
            return "The number of vectors and metadatas should be the same"

        ids = [str(uuid.uuid4()) for _ in vectors]
        vector_list = list(zip(ids, vectors, metadatas))

        batched_vectors = [vector_list[i:i+batch_size] for i in range(0, len(vector_list), batch_size)]

        if vdb_type == "Pinecone":
            client = PineconeClient(pinecone_api_key)
            index = client.Index(host = pinecone_host)            

            async_res = [
                index.upsert(
                    vectors=batch,
                    async_req=True
                )
                for batch in batched_vectors
            ]
            [res.get() for res in async_res]
            # index.upsert(vectors=vector_list, async_req=False)

            gr.Info("Data imported successfully")
            return index.describe_index_stats().to_str()
        elif vdb_type == "Milvus":
            client = MilvusClient(uri=milvus_uri, token=milvus_token)
            # convert vector_list to data like: [{"pk": 0, "vector": [0.3580376395471989, ...], "text": "pink_8682"},...]
            data = [{"vector": vector, "text": metadata[input_dataframe.columns[1]]} for i, (_, vector, metadata) in enumerate(vector_list)]
            client.insert(milvus_collection_name, data)
            gr.Info("Data imported successfully")
            return client.get_collection_stats(milvus_collection_name)
                    
        return "No VDB selected"

    def embed_import_dataframe(self,
                               input_dataframe,
                               key_field,
                               value_fields,
                               model_name,
                               vdb_type,
                               pinecone_host, pinecone_api_key,
                               milvus_uri, milvus_token, milvus_collection_name,
                               batch_size):
        _, embedded_vectors = self._embed_dataframe(
            input_dataframe,
            key_field, value_fields,
            model_name
        )

        # Don't update output dataframe, import the data directly
        return self.import_data(
            vdb_type,
            pinecone_host, pinecone_api_key,
            milvus_uri, milvus_token, milvus_collection_name,
            input_dataframe, embedded_vectors, batch_size
        )

    def vdb_search(
            self,
            vdb_type,
            pinecone_host,
            pinecone_api_key,
            milvus_uri,
            milvus_token,
            milvus_collection_name,
            embedding
    ):
        # example of embedding: "[.1,.2,.5,...]"
        # convert string embedding to a list of float
        vector = [float(i) for i in embedding[1:-1].split(',')] if isinstance(embedding, str) else embedding

        if vdb_type == "Pinecone":
            client = PineconeClient(pinecone_api_key)
            index = client.Index(host=pinecone_host)
            result = index.query(vector=vector, top_k = 3, include_metadata=True)
            return result, ''
        elif vdb_type == "Milvus":
            client = MilvusClient(uri=milvus_uri, token=milvus_token)
            result = client.search(
                collection_name=milvus_collection_name,
                data=[vector],
                limit=3
            )
            return result, ''
        
    def embed_search(
            self,
            model_name,
            vdb_type,
            pinecone_host,
            pinecone_api_key,
            milvus_uri,
            milvus_token,
            milvus_collection_name,
            input_text
    ):
        # model_index = next((index for index, setting in enumerate(model_settings) if setting['model'] == model_name), 0)
        # model = OpenAIEmbeddings(**model_settings[model_index])

        model = self._create_embedding_model(model_name)
        embedding = model.embed_query(input_text)

        return self.vdb_search(
            vdb_type,
            pinecone_host,
            pinecone_api_key,
            milvus_uri,
            milvus_token,
            milvus_collection_name,
            embedding
        )

    def cluster_dataframe(self, input_dataframe, embeddings_file, clusters):
        # csv = pd.read_csv(embeddings_file).iloc[:, 0].tolist()
        # # Convert the array of strings into ndarray
        # data = np.array([np.array(ast.literal_eval(row)) for row in csv])

        data = np.load(embeddings_file)

        if len(data) < clusters:
            gr.Warning("The number of clusters should be less than the number of data points")
            return input_dataframe, None

        # Apply Spectral Clustering
        spectral_clustering = SpectralClustering(n_clusters=clusters, random_state=0).fit(data)

        # Append the cluster labels to the last column of the dataframe
        input_dataframe['Cluster'] = spectral_clustering.labels_

        temp_filename = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        input_dataframe.to_csv(temp_filename, index=False)

        gr.Info(f"Data clustered successfully into {clusters} clusters")

        return input_dataframe, temp_filename

if __name__ == "__main__":
    ui = EmbeddingUI()
    ui.ui.launch(share=False)