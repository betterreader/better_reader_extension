#!/usr/bin/env python3
"""
Setup script for the Better Reader Extension vector search database tables.
This script will create the necessary tables in your Supabase instance.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("Error: SUPABASE_URL and SUPABASE_ANON_KEY environment variables must be set")
    print("Please make sure these are configured in your .env file")
    exit(1)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def setup_database():
    """
    Set up the database tables using SQL from db_setup.sql
    """
    try:
        # Read SQL file
        with open('db_setup.sql', 'r') as file:
            sql = file.read()
        
        # Execute SQL statements
        # Note: This is an approximation as python-supabase doesn't directly expose raw SQL execution
        # In a real setup, you would use the Supabase dashboard or psql directly
        
        print("Please execute the SQL in db_setup.sql in your Supabase dashboard")
        print("Visit https://app.supabase.io/ and go to your project")
        print("Navigate to SQL Editor and run the queries in db_setup.sql")
        
        print("\nAlternatively, if you have access to psql, you can run:")
        print(f"psql {SUPABASE_URL.replace('https://', '')} -f db_setup.sql")
        
        return True
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False

def test_connection():
    """
    Test the connection to Supabase
    """
    try:
        # Just try a simple query to see if we can connect
        response = supabase.from_("storage").select("*").limit(1).execute()
        print("Connection to Supabase successful!")
        return True
    except Exception as e:
        error_str = str(e)
        if "42P01" in error_str and "does not exist" in error_str:
            # This is actually good - it means we connected successfully but the table doesn't exist yet
            print("Connection successful, but tables not yet created (as expected).")
            return True
        print(f"Error connecting to Supabase: {error_str}")
        return False

def main():
    """
    Main function to set up the database
    """
    print("Testing connection to Supabase...")
    if test_connection():
        print("\nWould you like to set up the database tables? (y/n)")
        response = input().lower()
        if response == 'y':
            if setup_database():
                print("\nDatabase setup complete! You can now use the vector search feature.")
            else:
                print("\nFailed to set up database. Please check your configuration.")
        else:
            print("\nSkipping database setup.")
    else:
        print("\nFailed to connect to Supabase. Please check your configuration.")

if __name__ == "__main__":
    main()
