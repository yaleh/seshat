import gradio as gr
import numpy as np
import pandas as pd
import tempfile
import uuid
import ast
import json
from pinecone import Pinecone as PineconeClient
from sklearn.cluster import SpectralClustering
from pymilvus import MilvusClient
from components.lcel import EmbeddingFactory
from tools.table_parser import TableParser
from tools.utils import detect_encoding
from db.db_sqlite3 import (
    DatabaseManager, EMBEDDING_PINECONE_HOSTS_TABLE,
    EMBEDDING_PINECONE_API_KEYS_TABLE, EMBEDDING_MILVUS_URIS_TABLE,
    EMBEDDING_MILVUS_TOKENS_TABLE, EMBEDDING_MILVUS_COLLECTIONS_TABLE
)

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

        self.db_manager = DatabaseManager(
            self.config.server.message_db, 
            self.config.server.max_message_length
        )

        self.vdb_settings = {
            "Pinecone": {
                "host": None,
                "api_key": None
            },
            "Milvus": {
                "uri": None,
                "token": None,
                "collection_name": None
            },
            "Chroma": {
                "filename": None
            }
        }
        self.current_vdb_type = "Pinecone"

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
                        ["Pinecone", "Milvus", "Chroma", "FAISS"],
                        label="Vector Database",
                        value="Pinecone",
                        visible=False,
                        interactive=True
                    )

                    self.pinecone_tab = gr.Tab("Pinecone")
                    self.milvus_tab = gr.Tab("Milvus")

                    # Todo:
                    # * Upload Chroma DB package in zip/tar.gz
                    # * Update the current DB
                    # * Download the DB package
                    self.chroma_tab = gr.Tab("Chroma")

                    with self.pinecone_tab:
                        self.pinecone_host = gr.Dropdown(
                            choices=self.db_manager.get_messages(EMBEDDING_PINECONE_HOSTS_TABLE),
                            label="Pinecone Host",
                            allow_custom_value=True,
                            interactive=True
                        )
                        self.pinecone_api_key = gr.Dropdown(
                            choices=self.db_manager.get_messages(EMBEDDING_PINECONE_API_KEYS_TABLE),
                            label="Pinecone API Key",
                            allow_custom_value=True,
                            interactive=True
                        )
                        pinecone_components = [self.pinecone_host, self.pinecone_api_key]
                        for component in pinecone_components:
                            component.change(
                                self.update_pinecone_settings,
                                pinecone_components,
                                []
                            )

                    with self.milvus_tab:
                        self.milvus_uri = gr.Dropdown(
                            choices=self.db_manager.get_messages(EMBEDDING_MILVUS_URIS_TABLE),
                            label="Milvus URI",
                            allow_custom_value=True,
                            interactive=True
                        )
                        self.milvus_token = gr.Dropdown(
                            choices=self.db_manager.get_messages(EMBEDDING_MILVUS_TOKENS_TABLE),
                            label="Milvus Token",
                            allow_custom_value=True,
                            interactive=True
                        )
                        self.milvus_collection_name = gr.Dropdown(
                            choices=self.db_manager.get_messages(EMBEDDING_MILVUS_COLLECTIONS_TABLE),
                            label="Milvus Collection Name",
                            allow_custom_value=True,
                            interactive=True
                        )
                        milvus_components = [self.milvus_uri, self.milvus_token, self.milvus_collection_name]
                        for component in milvus_components:
                            component.change(
                                self.update_milvus_settings,
                                milvus_components,
                                []
                            )

                    with self.chroma_tab:
                        with gr.Group():
                            self.chroma_file = gr.File(label="Chroma File", interactive=True)
                            self.save_chroma_btn = gr.Button(value="Save Chroma", variant='primary')
                        self.chroma_file.upload(
                            self.update_chroma_settings,
                            [self.chroma_file],
                            []
                        )
                        self.chroma_file.clear(
                            self.update_chroma_settings,
                            [self.chroma_file],
                            []
                        )

                    self.reload_vdb_settings_btn = gr.Button(value="Reload VDB Settings",
                                                             variant='secondary')

                    with gr.Group():
                        self.vdb_index_description = gr.Textbox(
                            label="VDB Index Description",
                            placeholder="VDB Index Description",
                            interactive=False
                        )
                        with gr.Tab('General'):
                            self.refresh_vdb_btn = gr.Button(value="Refresh", variant='primary')
                        with gr.Tab('Advanced'):
                            self.clear_vdb_btn = gr.Button(value="Clear VDB", variant='danger')

                with gr.Column(scale=3):
                    # Add a Gradio Dataframe and a Gradio File component
                    with gr.Tab("Table"):
                        self.input_dataframe = gr.Dataframe(
                            interactive=True,
                            wrap=True
                        )
                        with gr.Row():
                            with gr.Column(min_width=160):
                                with gr.Group():
                                    self.file_table = gr.File(label="Table Data")
                                    self.save_table_btn = gr.Button(value="Save Table", variant='primary')

                            with gr.Column(min_width=160):
                                with gr.Group():
                                    self.file_embeddings = gr.File(
                                        label="Embeddings",
                                        file_types=['csv', 'xls', 'xlsx']
                                    )
                                    self.import_data_btn = gr.Button(value="Import Data", variant='primary')

                            with gr.Column(min_width=160):
                                self.key_field = gr.Textbox(label="Key Field", value="id")
                                self.value_fields = gr.Textbox(
                                    label="Value Fields",
                                    placeholder="field1, field2, ...",
                                    value="text"
                                )
                            with gr.Column(min_width=160):
                                self.batch_start = gr.Number(label="Start", value=0, minimum=0, precision=0)
                                self.batch_end = gr.Number(label="End (excluded)", value=0, minimum=0, precision=0)
                                self.batch_size = gr.Number(
                                    label="Batch Size",
                                    value=1, minimum=0, precision=0,
                                    info="Batch for embedding/importing"
                                    )

                        with gr.Tab("Embedding"):
                            with gr.Row():
                                self.embed_dataframe_btn = gr.Button(value="Embed Dataframe", variant='primary')
                                self.embed_import_btn = gr.Button(value="Embed and Import Data", variant='primary')

                        with gr.Tab("Clustering"):
                            with gr.Row():
                                with gr.Column(min_width=160):
                                    self.recreate_cluster_output_checkbox = gr.Checkbox(label="Recreate Cluster Output", value=False)
                                    self.cluster_output_field = gr.Textbox(
                                        label = "Cluster Output Field",
                                        value = "Cluster",
                                        interactive=True
                                    )
                                with gr.Column(min_width=160):
                                    self.clusters = gr.Number(label="Clusters", value=10)
                                    self.cluster_label_base = gr.Number(label="Cluster Label Base", value=0)
                                with gr.Column(min_width=160):
                                    self.clustering_btn = gr.Button(value="Cluster Analysis", variant='primary')
                                    self.add_index_column_btn = gr.Button(value="Add Index Column", variant='secondary')

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
                        with gr.Row():
                            self.embed_text_btn = gr.Button(value="Embed", variant='primary')
                            self.clear_text_embedding_btn = gr.ClearButton(
                                [self.output_embedding],
                                value="Clear Tex Embedding"
                            )

                        with gr.Tab("Search"):
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
                                self.vdb_search_k_number = gr.Number(label="Top K", value=3)
                                self.vdb_search_btn = gr.Button(value="Search VDB", variant='primary')
                                self.embed_search_btn = gr.Button(value="Embed and Search VDB", variant='primary')
                                self.clear_text_all_btn = gr.ClearButton(
                                    [self.input_text, self.output_embedding, self.vdb_search_output, self.vdb_search_meta],
                                    value="Clear All"
                                )

                        with gr.Tab("Upsert"):
                            self.upsert_text = gr.TextArea(
                                label="Upsert Text",
                                placeholder="Upsert Text",
                                show_copy_button=True,
                                interactive=True
                            )
                            with gr.Row():
                                self.upsert_text_btn = gr.Button(value="Upsert", variant='primary')
                                self.overwrite_checkbox = gr.Checkbox(label="Overwrite", value=False)

                        with gr.Tab("Delete"):
                            self.id_to_delete_text = gr.Textbox(
                                label="ID to Delete",
                                placeholder="ID to Delete",
                                interactive=True
                            )
                            self.delete_text_btn = gr.Button(value="Delete", variant='danger')

            self.pinecone_tab.select(self.select_vdb_tab, [], [self.vdb_type])
            self.milvus_tab.select(self.select_vdb_tab, [], [self.vdb_type])
            self.chroma_tab.select(self.select_vdb_tab, [], [self.vdb_type])

            self.reload_vdb_settings_btn.click(
                self.reload_vdb_settings,
                [],
                [self.pinecone_host, self.pinecone_api_key, self.milvus_uri, self.milvus_token, self.milvus_collection_name]
            )                

            self.refresh_vdb_btn.click(
                self.refresh_vdb,
                [],
                [self.vdb_index_description]
            )
            self.clear_vdb_btn.click(
                self.clear_vdb,
                [],
                [self.vdb_index_description]
            )
            self.file_table.upload(
                self.upload_data,
                [self.file_table],
                [
                    self.input_dataframe,
                    self.key_field,
                    self.value_fields,
                    self.batch_start,
                    self.batch_end,
                    self.batch_size
                ]
            )
            self.embed_dataframe_btn.click(
                self.embed_dataframe,
                [
                    self.input_dataframe, 
                    self.key_field, 
                    self.value_fields,
                    self.model_name,
                    self.batch_start,
                    self.batch_end,
                    self.batch_size
                ],
                [self.file_embeddings]
            )
            self.import_data_btn.click(
                self.import_data,
                [
                    self.input_dataframe,
                    self.value_fields,
                    self.file_embeddings,
                    self.batch_start,
                    self.batch_end,
                    self.batch_size
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
                    self.batch_start,
                    self.batch_end,
                    self.batch_size
                ],
                [self.vdb_index_description]
            )
            self.clustering_btn.click(
                self.cluster_dataframe,
                [
                    self.input_dataframe, self.file_embeddings,
                    self.recreate_cluster_output_checkbox, self.cluster_output_field,
                    self.clusters, self.cluster_label_base,
                    self.batch_start, self.batch_end
                ],
                [self.input_dataframe, self.file_table]
            )

            self.add_index_column_btn.click(
                self.add_index_column,
                [self.input_dataframe],
                [self.input_dataframe]
            )

            self.embed_text_btn.click(
                self.embed_text, 
                [self.input_text, self.model_name], 
                [self.output_embedding]
            )

            self.vdb_search_btn.click(
                self.vdb_search,
                [
                    self.output_embedding,
                    self.vdb_search_k_number
                ],
                [self.vdb_search_output, self.vdb_search_meta]
            )
            self.embed_search_btn.click(
                self.embed_search,
                [
                    self.model_name,
                    self.input_text,
                    self.vdb_search_k_number
                ],
                [self.vdb_search_output, self.vdb_search_meta]
            )
            self.upsert_text_btn.click(
                self.upsert_embeded_text,
                [
                    self.output_embedding, 
                    self.upsert_text, 
                    self.overwrite_checkbox
                ],
                [self.vdb_index_description]
            )
            self.delete_text_btn.click(
                self.delete_embedded_text,
                [
                    self.id_to_delete_text
                ],
                [self.vdb_index_description]
            )
            self.save_table_btn.click(
                self.save_table,
                [self.input_dataframe],
                [self.file_table]
            )

        return block
    
    def select_vdb_tab(self, event: gr.SelectData):
        self.current_vdb_type = event.value
        return str(event.value)           
    
    def embed_text(self, input_text, model_name):
        try:
            model = self._create_embedding_model(model_name)
            vector = model.embed_query(input_text)
            # TODO: remove '[]' from the begining and end of the output
            return vector
        except Exception as e:
            raise gr.Error(f'Error: {e}')
        
    def update_vdb_history_db(self):
        if self.current_vdb_type == "Pinecone":
            self.db_manager.append_message(EMBEDDING_PINECONE_HOSTS_TABLE, self.vdb_settings["Pinecone"]["host"])
            self.db_manager.append_message(EMBEDDING_PINECONE_API_KEYS_TABLE, self.vdb_settings["Pinecone"]["api_key"])
        elif self.current_vdb_type == "Milvus":
            self.db_manager.append_message(EMBEDDING_MILVUS_URIS_TABLE, self.vdb_settings["Milvus"]["uri"])
            self.db_manager.append_message(EMBEDDING_MILVUS_TOKENS_TABLE, self.vdb_settings["Milvus"]["token"])
            self.db_manager.append_message(EMBEDDING_MILVUS_COLLECTIONS_TABLE, self.vdb_settings["Milvus"]["collection_name"])

    def reload_vdb_settings(self):
        return (
            gr.update(
                choices=self.db_manager.get_messages(EMBEDDING_PINECONE_HOSTS_TABLE) or None
            ),
            gr.update(
                choices=self.db_manager.get_messages(EMBEDDING_PINECONE_API_KEYS_TABLE) or None
            ),
            gr.update(
                choices=self.db_manager.get_messages(EMBEDDING_MILVUS_URIS_TABLE) or None
            ),
            gr.update(
                choices=self.db_manager.get_messages(EMBEDDING_MILVUS_TOKENS_TABLE) or None
            ),
            gr.update(
                choices=self.db_manager.get_messages(EMBEDDING_MILVUS_COLLECTIONS_TABLE) or None
            )
        )

    def refresh_vdb(self):
        self.update_vdb_history_db()
        if self.current_vdb_type == "Pinecone":
            client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
            index = client.Index(host=self.vdb_settings["Pinecone"]["host"])
            return index.describe_index_stats().to_str()
        elif self.current_vdb_type == "Milvus":
            milvus = MilvusClient(
                uri=self.vdb_settings['Milvus']['uri'],
                token=self.vdb_settings['Milvus']['token']
            )
            return milvus.get_collection_stats(self.vdb_settings['Milvus']['collection_name'])

        return "No VDB selected"

    def clear_vdb(self):
        self.update_vdb_history_db()
        if self.current_vdb_type == "Pinecone":
            client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
            index = client.Index(host=self.vdb_settings["Pinecone"]["host"])
            index.delete(delete_all=True)
            gr.Info("VDB index cleared successfully")
            return index.describe_index_stats().to_str()
        elif self.current_vdb_type == "Milvus":
            client = MilvusClient(uri=self.vdb_settings["Milvus"]["uri"], token=self.vdb_settings["Milvus"]["token"])

            # get all ids
            ids = client.query(self.vdb_settings["Milvus"]["collection_name"], filter="id >= 0", output_fields=["id"])
            # ids looks like "[{'id': 0}, {'id': 1}, ...]"
            if ids is not None and len(ids) > 0:
                ids = [int(id['id']) for id in ids]
                # delete all ids
                client.delete(self.vdb_settings["Milvus"]["collection_name"], ids)

                gr.Info("VDB index cleared successfully")
            else:
                gr.Info("No data to delete")
            return client.get_collection_stats(self.vdb_settings["Milvus"]["collection_name"])
        
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

            # get the name of the first column
            key_field = df.columns[0] if len(df.columns) > 0 else "id"
            value_fields = ", ".join(df.columns[1:]) if len(df.columns) > 1 else "text"

            # get the rows number
            rows = len(df)
        except Exception as e:
            raise gr.Error(e)
        return (
            df, key_field, value_fields,
            gr.update(value=0, minimum=0, maximum=rows),
            gr.update(value=rows, minimum=0, maximum=rows),
            gr.update(value=1, minimum=0, maximum=rows)
        )
    
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

    def _embed_dataframe(self, input_dataframe, key_field, value_fields, model_name, 
                         start, end, batch_size):
        model = self._create_embedding_model(model_name)

        # split the value_fields into a list and remove any leading or trailing whitespaces
        value_fields = [field.strip() for field in value_fields.split(",")]

        embedded_vectors = []

        key_column = input_dataframe[key_field][start:end]
        # value_columns = input_dataframe[value_fields]

        embedded_vectors = []
        for i in range(0, len(key_column), batch_size):
            batch_keys = key_column[i:i+batch_size]
            batch_list = batch_keys.tolist()
            # model.embed_documents doesn't accept empty strings or None
            # fill the empty strings with a space
            batch_list = [key if key != "" else " " for key in batch_list]
            batch_vectors = model.embed_documents(batch_list)
            embedded_vectors.extend(batch_vectors)

        # create a new dataframe with the embedded_vectors (string) as the firt column named `Vector`
        embedded_table = pd.DataFrame({"Vector": embedded_vectors})

        # append value_fields to the right of the new dataframe
        embedded_table = pd.concat([embedded_table, input_dataframe[value_fields][start:end]], axis=1)

        return embedded_table, embedded_vectors

    def embed_dataframe(self, input_dataframe, key_field, value_fields, model_name,
                        start, end, batch_size):
        try:
            # only vectors are saved
            # embedded_table is dropped
            _, embedded_vectors = self._embed_dataframe(
                input_dataframe,
                key_field,
                value_fields,
                model_name,
                start,
                end,
                batch_size
                )

            # save embedded vectors to a temporary npy file
            temp_filename = tempfile.NamedTemporaryFile(suffix=".npy", delete=False).name
            np.save(temp_filename, embedded_vectors)

            return temp_filename
        except Exception as e:
            gr.Warning(f'Error: {e}')
            return None        
 
    def import_data(self,
                    input_dataframe, value_fields, embeddings,
                    start, end, batch_size):
        self.update_vdb_history_db()
        value_fields = [field.strip() for field in value_fields.split(",")]
        metadatas = input_dataframe[value_fields][start:end].to_dict(orient='records')
        if isinstance(embeddings, gr.File) or isinstance(embeddings, str):
            vectors = np.load(embeddings)
        else:
            vectors = embeddings

        # validate the number of vectors and metadatas
        if len(vectors) != len(metadatas):
            gr.Warning("The number of vectors and metadatas should be the same")
            return "The number of vectors and metadatas should be the same"

        if self.current_vdb_type == "Pinecone":
            ids = [str(uuid.uuid4()) for _ in vectors]
            vector_list = list(zip(ids, vectors, metadatas))

            batched_vectors = [vector_list[i:i+batch_size] for i in range(0, len(vector_list), batch_size)]

            client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
            index = client.Index(host = self.vdb_settings["Pinecone"]["host"])            

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
        elif self.current_vdb_type == "Milvus":
            # vectors is a arry like [[0.3580376395471989, ...], ...]
            # metadatas is a list of dictionaries like [{"text": "pink_8682"},...]
            # create vector_list like [{"vector": [0.3580376395471989, ...], "text": "pink_8682"},...]
            vector_list = [{"vector": vector, **{field: metadata[field] for field in value_fields}} for vector, metadata in zip(vectors, metadatas)]
            client = MilvusClient(
                uri=self.vdb_settings["Milvus"]["uri"],
                token=self.vdb_settings["Milvus"]["token"]
            )
            client.insert(self.vdb_settings["Milvus"]["collection_name"], vector_list)
            gr.Info("Data imported successfully")
            return client.get_collection_stats(self.vdb_settings["Milvus"]["collection_name"])
                    
        return "No VDB selected"
    
    def upsert_embeded_text(
            self, embedding, upsert_text, overwrite
            ):
        self.update_vdb_history_db()
        # convert embedding to a list of float
        vector = [float(i) for i in embedding[1:-1].split(',')] if isinstance(embedding, str) else embedding

        if self.current_vdb_type == "Pinecone":
            client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
            index = client.Index(host=self.vdb_settings["Pinecone"]["host"])

            index.upsert(vectors=[(str(uuid.uuid4()), vector, {"text": upsert_text})], overwrite=overwrite)
            gr.Info("Data upserted successfully")
            return index.describe_index_stats().to_str()
        elif self.current_vdb_type == "Milvus":
            client = MilvusClient(uri=self.vdb_settings["Milvus"]["uri"], token=self.vdb_settings["Milvus"]["token"])

            data = [{"vector": vector, "text": upsert_text}]
            client.insert(self.vdb_settings["Milvus"]["collection_name"], data)
            gr.Info("Data upserted successfully")
            return client.get_collection_stats(self.vdb_settings["Milvus"]["collection_name"])

    def delete_embedded_text(
            self, id_to_delete_text
            ):
        self.update_vdb_history_db()
        if self.current_vdb_type == "Pinecone":
            client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
            index = client.Index(host=self.vdb_settings["Pinecone"]["host"])

            index.delete(ids=[id_to_delete_text])
            gr.Info("Data deleted successfully")
            return index.describe_index_stats().to_str()
        elif self.current_vdb_type == "Milvus":
            client = MilvusClient(uri=self.vdb_settings["Milvus"]["uri"], token=self.vdb_settings["Milvus"]["token"])

            client.delete(self.vdb_settings["Milvus"]["collection_name"], [id_to_delete_text])
            gr.Info("Data deleted successfully")
            return client.get_collection_stats(self.vdb_settings["Milvus"]["collection_name"])

    def embed_import_dataframe(self,
                               input_dataframe,
                               key_field,
                               value_fields,
                               model_name,
                               start, end, batch_size):
        self.update_vdb_history_db()
        _, embedded_vectors = self._embed_dataframe(
            input_dataframe,
            key_field,
            value_fields,
            model_name,
            start,
            end,
            batch_size
        )

        # Don't update output dataframe, import the data directly
        return self.import_data(
            input_dataframe, value_fields, embedded_vectors,
            start, end, batch_size
        )

    def vdb_search(self, embedding, k):
        self.update_vdb_history_db()
        try:
            vector = [float(i) for i in embedding[1:-1].split(',')] if isinstance(embedding, str) else embedding

            if self.current_vdb_type == "Pinecone":
                client = PineconeClient(self.vdb_settings["Pinecone"]["api_key"])
                index = client.Index(host=self.vdb_settings["Pinecone"]["host"])
                result = index.query(vector=vector, top_k = k, include_metadata=True)
                result = result.to_str()

            elif self.current_vdb_type == "Milvus":
                client = MilvusClient(uri=self.vdb_settings["Milvus"]["uri"], token=self.vdb_settings["Milvus"]["token"])
                result = client.search(
                    collection_name=self.vdb_settings["Milvus"]["collection_name"],
                    data=[vector],
                    output_fields=["text"],
                    limit=k
                )
                # beautify the json result
                result = json.dumps(result, indent=4, ensure_ascii=False).encode('utf8').decode()
        except Exception as e:
            raise gr.Error(f'Error: {e}')
                    
        return result, ''
        
    def embed_search(
            self,
            model_name,
            input_text,
            k
    ):
        try:
            model = self._create_embedding_model(model_name)
            embedding = model.embed_query(input_text)

            return self.vdb_search(
                embedding,
                k
            )
        except Exception as e:
            raise gr.Error(f'Error: {e}')

    def cluster_dataframe(
            self, input_dataframe, embeddings_file,
            recreate_cluster_output, cluster_output_field,
            clusters, cluster_label_base,
            start, end):

        # raise an error if input_dataframe or embeddings_file is empty
        if input_dataframe.shape[0]<=1 or embeddings_file is None:
            gr.Warning("Input dataframe or embeddings file is empty")
            return input_dataframe, None
            # return input_dataframe

        data = np.load(embeddings_file)

        if len(data) < clusters:
            gr.Warning("The number of clusters should be less than the number of data points")
            return input_dataframe, None
            # return input_dataframe

        # validate data length vs start&end
        if len(data) != end - start:
            gr.Warning("The number of vectors and data points should be the same")
            return input_dataframe, None
            # return input_dataframe

        try:
            # Apply Spectral Clustering
            spectral_clustering = SpectralClustering(n_clusters=clusters, random_state=0).fit(data)

            # add cluster label base to the cluster labels
            spectral_clustering.labels_ += cluster_label_base

            if recreate_cluster_output:
                # drop the cluster output field if it exists
                input_dataframe = input_dataframe.drop(columns=[cluster_output_field], errors='ignore')
            # create an empty cluster output column if it doesn't exist
            if cluster_output_field not in input_dataframe.columns:
                input_dataframe[cluster_output_field] = None

            # overwrite the rows (specified by start and end) of cluster output field if it exists
            input_dataframe.loc[start:end-1, cluster_output_field] = spectral_clustering.labels_

            temp_filename = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
            input_dataframe.to_csv(temp_filename, index=False)

            gr.Info(f"Data clustered successfully into {clusters} clusters")

            return input_dataframe, temp_filename
            # return input_dataframe
        except Exception as e:
            raise gr.Error(f"Error: {e}")

    def add_index_column(self, input_dataframe):
        if 'index' not in input_dataframe.columns:
            input_dataframe.insert(0, 'index', range(0, len(input_dataframe)))

        # fill the index column with the row number if it's empty
        input_dataframe['index'] = range(len(input_dataframe))

        return input_dataframe
    
    def save_table(self, input_dataframe):
        temp_filename = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        input_dataframe.to_csv(temp_filename, index=False)
        return temp_filename

    def update_pinecone_settings(self, pinecone_host, pinecone_api_key):
        # create self.vdb_settings["Pinecone"] if it doesn't exist
        if "Pinecone" not in self.vdb_settings:
            self.vdb_settings["Pinecone"] = {}
        self.vdb_settings["Pinecone"]["host"] = pinecone_host
        self.vdb_settings["Pinecone"]["api_key"] = pinecone_api_key

    def update_milvus_settings(self, milvus_uri, milvus_token, milvus_collection_name):
        # create self.vdb_settings["Milvus"] if it doesn't exist
        if "Milvus" not in self.vdb_settings:
            self.vdb_settings["Milvus"] = {}
        self.vdb_settings["Milvus"]["uri"] = milvus_uri
        self.vdb_settings["Milvus"]["token"] = milvus_token
        self.vdb_settings["Milvus"]["collection_name"] = milvus_collection_name

    def update_chroma_settings(self, file):
        # create self.vdb_settings["Chroma"] if it doesn't exist
        if "Chroma" not in self.vdb_settings:
            self.vdb_settings["Chroma"] = {}
        if file is None:
            self.vdb_settings["Chroma"]["file"] = None
        else:
            self.vdb_settings["Chroma"]["file"] = file.name

if __name__ == "__main__":
    ui = EmbeddingUI()
    ui.ui.launch(share=False)