<h1>Embeddings API</h1>
<h3>Vector Embedding & Database API utilizing Cloudflare Workers AI, Vectorize or Qdrant, and Cloudflare D1 (optional)</h3>


## Features
- **Distributed**
- **Scalable**
- **Simple endpoints with basic validation**
- **Basic authentication**
- **Support for vector paging when using Cloudflare Vectorize with D1**
- **OpenAPI support**

## API Endpoints
### `GET /api/v1/embeddings/cloudflare/{namespace}`
Retrieves all embeddings for a given Cloudflare namespace ([Vectorize Index](https://developers.cloudflare.com/api/operations/vectorize-list-vectorize-indexes)). 
Only supported if a valid `CLOUDFLARE_D1_DATABASE_IDENTIFIER` has been set 
in the project's environment variables. 

If you haven't already created a **Cloudflare D1** database and wish to utilize
this API endpoint, you will first need to create a database in your Cloudflare console and then provide
the database identifier as outlined above. 

Note that **Cloudflare D1** currently only supports databases up to 2GB in size,
so this may not be suitable for production use, depending on your anticipated
data volume. Click [here](https://developers.cloudflare.com/d1/platform/limits) for up-to-date D1 limitations.

### `GET /api/v1/embeddings/cloudflare/{namespace}/{embedding_id}`
Retrieves a specific vector from the Cloudflare namespace for with a given `id`.

### `POST /api/v1/embeddings/cloudflare/{namespace}`
Create and persist an embedding vector using Cloudflare Workers AI (text embedding models)[https://developers.cloudflare.com/workers-ai/models/#text-embeddings].
