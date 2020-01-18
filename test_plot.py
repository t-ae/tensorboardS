from tensorboardX import SummaryWriter

logdir = "./logdir"
writer = SummaryWriter(logdir)

writer.add_text("tag", "hoge", walltime=0)
writer.flush()
writer.close()

