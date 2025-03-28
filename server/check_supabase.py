import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase credentials not found in environment variables")
    print(f"Available environment variables: {[k for k in os.environ.keys() if 'SUPABASE' in k]}")
    exit(1)

print(f"Supabase URL: {SUPABASE_URL}")
print(f"Supabase Key: {SUPABASE_KEY[:4]}...{SUPABASE_KEY[-4:]}")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

try:
    # Check connection by querying the articles table
    response = supabase.table('articles').select('*').limit(10).execute()
    
    # Print the response for debugging
    print(f"\nArticles count: {len(response.data) if hasattr(response, 'data') else 0}")
    
    if hasattr(response, 'data') and response.data:
        print("\nArticles found:")
        for i, article in enumerate(response.data):
            print(f"\n--- Article {i+1} ---")
            print(f"ID: {article.get('id')}")
            print(f"Title: {article.get('title')}")
            print(f"URL: {article.get('url')}")
            print(f"Summary: {article.get('summary', 'No summary')[:100]}..." if article.get('summary') else "No summary")
            print(f"Topics: {article.get('topics', [])}")
            print(f"Created at: {article.get('created_at')}")
    else:
        print("\nNo articles found in the database")
    
    # Try inserting a test article
    print("\nInserting a test article...")
    test_article = {
        'id': 'test-article-id',
        'url': 'https://example.com/test-article',
        'title': 'Test Article from Check Script',
        'user_id': None,
        'topics': ['Test', 'Debug', 'Supabase'],
        'summary': 'This is a test article created by the check_supabase.py script to verify database connectivity.',
        'created_at': '2025-03-27T19:15:00.000000'
    }
    
    insert_response = supabase.table('articles').insert(test_article).execute()
    print(f"Insert response: {insert_response}")
    
    # Now try to retrieve the test article
    print("\nRetrieving the test article...")
    test_response = supabase.table('articles').select('*').eq('id', 'test-article-id').execute()
    
    if hasattr(test_response, 'data') and test_response.data:
        print("Test article successfully retrieved!")
        print(f"Test article data: {test_response.data[0]}")
    else:
        print("Failed to retrieve the test article")
    
except Exception as e:
    print(f"\nError connecting to Supabase: {str(e)}")
    import traceback
    traceback.print_exc()
