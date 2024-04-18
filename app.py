from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from supabase import create_client
from awsUtils import delete_from_s3, upload_to_s3
from pineconeUtils import delete_from_pinecone, upload_to_pinecone, load_s3_files
from rag import similarity_search_for_documents
from rag import get_rachel_answer
from initiial_evaluate import evaluate, evaluate2, load_pdfs
from testdataset import testdf
from flask_socketio import SocketIO
from flask_socketio import send, emit
from flask_cors import CORS, cross_origin


load_dotenv()

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

supabase = create_client(url, key)
app = Flask(__name__)

cors = CORS(app)

socketio = SocketIO(app, cors_allowed_origins='*')


@socketio.on('rachel')
def handle_message(data):
    try:
        if 'question' not in data or 'pinecone_index' not in data:
            emit('rachel', {
                'error': 'Please provide both question and pinecone_index'}, broadcast=True)
            return
        question = data['question']
        pinecone_index = data['pinecone_index']

        answer_generator = get_rachel_answer(
            question, pinecone_index)

        answer_generating = True
        answer_string = ''
        while answer_generating:
            try:
                next_answer = next(answer_generator)
                answer_string += next_answer
                emit('rachel', {"answer": answer_string,
                     "done": False}, broadcast=True)
                print(next_answer, '')
            except Exception as e:
                answer_generating = False
                emit('rachel', {"answer": answer_string,
                     "done": True}, broadcast=True)

                print('')

    except Exception as e:
        emit('rachel', {'error': str(e)}, broadcast=True)
        return


# creates a user in the database


@app.route('/createUser', methods=['POST'])
def createUser():

    data = request.json

    if 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Please provide both username and password'}), 400
    username = data['username']
    password = data['password']

    try:
        supabase.table('users').insert(
            {'username': username, 'password': password}).execute()
        response = jsonify({"message": "User created successfully"}), 200
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# handles file upload to s3 and then pinecone


@app.route('/uploadFiles', methods=['POST'])
def uploadFiles():
    try:
        files = request.files.getlist("files")
        print(request.files)

        if (len(files) == 0):
            print('no files')
            return jsonify({'error': 'Please provide files to upload'}), 400
        # make sure files are pdfs
        for file in files:
            if file.filename[-3:] != 'pdf':
                return jsonify({'error': 'Please provide only pdf files'}), 400

        # upload to s3
        doc_names = upload_to_s3(files, 'avalondemobucket')

        # upload to supabase and upload to pinecone
        upload_to_pinecone('avalondemobucket', 'testuser', doc_names)

        response = jsonify({"success": "Successfully uploaded",
                           "message": "Files uploaded to S3 and Pinecone successfully"}), 200
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# delete files from s3 and pinecone


@app.route('/deleteFiles', methods=['POST'])
def deleteFiles():
    try:
        data = request.json
        if 'files' not in data:
            return jsonify({'error': 'Please provide files to delete'}), 400
        files = data['files']

        # delete from s3
        delete_from_s3(files, 'avalondemobucket')

        # delete from pinecone
        delete_from_pinecone(files, 'avalondemobucket')

        response = jsonify({"success": "Successfully deleted",
                           "message": "Files deleted from S3 and Pinecone successfully"}), 200
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# retrieves files from supabase

@app.route('/retrieveFiles', methods=['GET'])
def retrieveFiles():
    try:
        files, _ = supabase.table('files').select('*').execute()
        file_list = []
        print(files[1])
        for file in files[1]:
            file_list.append(
                {'file': file['file'], 'url': file['url'], 'timestamp': file['created_at'], 'id': file['id'], "summary": file['summary']})
        print(file_list)
        response = jsonify(file_list), 200
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# gets rachel answer


@socketio.on('getRachelAnswer')
def getRachelAnswer(data):
    try:
        data = request.json
        if 'question' not in data or 'pinecone_index' not in data:
            return jsonify({'error': 'Please provide both question and pinecone_index'}), 400
        question = data['question']
        pinecone_index = data['pinecone_index']

        rachel_answer = get_rachel_answer(question, pinecone_index)
        return jsonify({"answer": rachel_answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# similarity search for documents
@app.route('/similaritySearch', methods=['POST'])
def similaritySearch():
    try:
        data = request.json
        query = data['query']
        index_name = data['index_name']
        k = data['k']
        response = similarity_search_for_documents(query, index_name, k)
        documents = []
        for doc in response:
            append_dict = {}
            append_dict['document'] = doc.page_content
            append_dict['metadata'] = doc.metadata
            documents.append(append_dict)
        print(response)
        return jsonify({"success": "Successfully retrieved documents", "documents": documents}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# evaluates the model
@app.route('/evaluateModel', methods=['POST'])
def evaluateModel():
    try:
        data = request.json
        pinecone_index = data['pinecone_index']
        docs = load_pdfs()
        print(docs)
        description = data.get("description")

        evaluation = evaluate2(pinecone_index, testdf, description)
        return jsonify({"evaluation": evaluation}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# get url from id in supabase
@app.route('/getUrlFromId', methods=['POST'])
def getUrl():
    try:
        data = request.json
        id = data['id']
        file, _ = supabase.table('files').select('*').eq('id', id).execute()
        return jsonify({"url": file[1][0]['url']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
    socketio.run(app)
