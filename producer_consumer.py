import torch, os, random, time
from torch.multiprocessing import JoinableQueue, Process, Lock, Queue
class Model(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.linear = torch.nn.Linear(3, 1)#.to(self.device)
        self.optim = torch.optim.Adam(self.parameters())
        self.producer_process_num = 4
        
    def forward(self, x):
        x = self.linear(x)
        return x.relu()
    
    def stageOne(self):
        self.linear.to(self.device)
        queue, lock = JoinableQueue(), Lock()
        resultqueue = Queue()
        producers, consumers = [], []
        values = torch.randint(1, 100, size=(100, ))
        for i in range(self.producer_process_num):
            producers.append(Process(target=self.producer, args=(queue, lock, values[i*25:(i+1)*25])))
        for j in range(2):
            p = Process(target=self.consumer, args=(queue, lock, resultqueue), daemon=True)
            consumers.append(p)   

        for p in producers:
            p.start()
        for c in consumers:
            c.start()

        #for c in consumers:
        #    c.join()    
        for p in producers:
            p.join()
        count=0
        while True:
            if resultqueue.empty:break
            loss=resultqueue.get()
            print(loss)
            #count+=1
            #if count==100:break
            #resultqueue.task_done()
        
    
    def producer(self, queue, lock, values):
        with lock:
            print(f'Starting Producer: {os.getpid()}')
        put_count=0
        for value in values:
            rands = torch.randint(low=0, high=value, size=(value,)).float()
            mean = torch.mean(rands)
            sm = torch.sum(rands)
            queue.put(torch.tensor([value, mean, sm]))
            #with lock:
            #    queue.put(torch.tensor([value, mean, sm]))
            #    print(f'Producer {os.getpid()} putted {put_count}')
            #    put_count+=1
        #with lock:
        #    queue.put(None)
        queue.join()
        with lock:
            print(f'Producer {os.getpid()}, exiting')
            
    def consumer(self, queue, lock, resultqueue):
        with lock:
            print(f'Starting Consumer {os.getpid()}')
        count=0
        get_count=0
        while True:
            x = queue.get()
            get_count+=1
            print('get count', get_count)
            loss_fn = torch.nn.MSELoss()
            self.optim.zero_grad()
            x = x.to(self.device)
            out = self.linear(x)
            y = torch.randint(0, 100, size=(1,)).float().to(self.device)
            loss = loss_fn(y.unsqueeze(1), out.unsqueeze(1))
            loss.backward()
            resultqueue.put(loss.cpu().detach())
            self.optim.step()
            print('x', x, 'out', out)
            queue.task_done()
        print('consumer exiting')
            
        '''
            time.sleep(random.uniform(0, 0.1))
            if queue.qsize==0:
                continue
            else:
                try:
                    with lock:
                        x = queue.get()
                        if x==None:
                            count+=1
                            if count==4:
                                print('None reached 4')
                                break
                        else:
                            x = x.to(self.device)
                            #print('x', x)
                            #x = x.to(self.device)
                            get_count+=1
                            print('get count', get_count)
                            #print(f'{os.getpid()} got {str(x)}')
                            #model = self.model.to('cuda:1')
                            loss_fn = torch.nn.MSELoss()
                            #optim.zero_grad()
                            out = self.linear(x)
                            y = torch.randint(0, 100, size=(1,)).float().to(self.device)
                            loss = loss_fn(y.unsqueeze(1), out.unsqueeze(1))
                            loss.backward()
                            optim.step()
                            print('x', x, 'out', out)
                except FileNotFoundError:
                    time.sleep(0.1)
        '''
        
def producer(queue, lock, values):
    with lock:
        print(f'Starting Producer: {os.getpid()}')
        
    for value in values:
        rands = torch.randint(low=0, high=value, size=(value,)).float()
        mean = torch.mean(rands)
        sm = torch.sum(rands)
        with lock:
            queue.put(torch.tensor([value, mean, sm]))
    queue.put(None)
    time.sleep(5)
    with lock:
        print(f'Producer {os.getpid()}, exiting')
        
def consumer(queue, lock, model, optim):
    with lock:
        print(f'Starting Consumer {os.getpid()}')
    count=0
    while True:
        #time.sleep(random.uniform(0, 0.1))
        if queue.qsize==0:
            continue
        else:
            with lock:
                x = queue.get()
            if x==None:
                count+=1
                if count==4:
                    break
            else:
                x = x.to('cuda:1')
                #print('x', x)
                #x = x.to(self.device)
                #print(f'{os.getpid()} got {str(x)}')
                model = model.to('cuda:1')
                #loss_fn = torch.nn.MSELoss()
                #optim.zero_grad()
                out = self.linear(x)
                #y = torch.randint(0, 100, size=(1,)).float().to('cuda:1')
                #loss = loss_fn(y.unsqueeze(1), out.unsqueeze(1))
                #loss.backward()
                #optim.step()
                print('x', x, 'out', out)
            
    print('Consumer exiting')
