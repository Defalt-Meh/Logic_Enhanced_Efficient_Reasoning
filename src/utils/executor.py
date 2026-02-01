import sys
import io
import contextlib

def execute_model_code(code_str):
    """
    EXECUTES THE TEXT AS CODE. 
    TEXT IS JUST BYTES WAITING TO BE INSTRUCTIONS.
    """
    # STRIP THE MARKDOWN BLOAT. 
    # LLMS LOVE TO ADD ``` LIKE THEY ARE WRITING A BLOG POST.
    # WE WANT RAW INSTRUCTIONS.
    clean_code = code_str.replace("```python", "").replace("```", "").strip()
    
    # WE CAPTURE STDOUT. LIKE TRAPPING A DEMON IN A JAR.
    output_capture = io.StringIO()
    
    try:
        # REDIRECT STDOUT. SILENCE THE CONSOLE. LISTEN ONLY TO THE CAPTURE.
        with contextlib.redirect_stdout(output_capture):
            # EMPTY SCOPE. NO GLOBAL VARIABLES. 
            # DONT LET IT TOUCH MY SYSTEM FILES OR I WILL RUN IT OVER WITH MY CAR.
            local_scope = {}
            exec(clean_code, {}, local_scope)
            
        # GOD'S TRUTH IS IN THE PRINT STATEMENTS.
        result = output_capture.getvalue().strip()
        return {"success": True, "output": result, "error": None}
        
    except Exception as e:
        # THE CODE WAS IMPERFECT. IT CRASHED. 
        # PROBABLY A SEGFAULT IN SPIRIT.
        return {"success": False, "output": None, "error": str(e)}