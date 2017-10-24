def print_progress(curr_progress, passed, total):
    progress = (passed / total) * 100 // 10
    if progress != curr_progress:
        print('.', sep=' ', end='', flush=True)
    return progress

def print_progess_end():
    print('!', sep=' ', flush=True)