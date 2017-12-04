def print_progress(curr_progress, passed, total):
    progress = min(int((passed / total) * 100 // 10), 10)
    if progress != curr_progress:
        for i in range(progress - curr_progress):
            print('.', sep=' ', end='', flush=True)
    return progress

def print_progess_end():
    print('!', sep=' ', flush=True)