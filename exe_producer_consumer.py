import torch
from torch.multiprocessing import JoinableQueue, Lock, Process, Queue

from producer_consumer import Model, consumer, producer

if __name__=='__main__':
    if torch.multiprocessing.get_start_method() == 'fork':
        torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    values = torch.randint(1, 100, size=(100, ))
    queue = JoinableQueue()
    resultqueue = JoinableQueue()
    lock = Lock()
    device = 'cuda:4'
    model = Model(device)
    model.to(device)
    model.stageOne()
    exit()
    optim = torch.optim.Adam(model.parameters())
    producers, consumers = [], []
    for i in range(4):
        producers.append(Process(target=model.producer, args=(queue, lock, values[i*25:(i+1)*25])))
    for j in range(1):
        p = Process(target=model.consumer, args=(queue, lock, model, optim), daemon=True)
        consumers.append(p)   

    for p in producers:
        p.start()
    for c in consumers:
        c.start()

    #for c in consumers:
    #    c.join()    
    for p in producers:
        p.join()
    
        
  
    print('Parent Process Exiting..')