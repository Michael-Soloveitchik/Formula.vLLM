# PR #8868: [Core] Avoid metrics log noise when idle

- field: scheduling  | area: engine/metrics.py
- author: russellb
- state: MERGED  created: 2024-09-26T20:58:18Z  merged: 2024-11-19T21:05:25Z
- url: https://github.com/vllm-project/vllm/pull/8868
- labels: ready

## Body


1a789ae7 [Core] Avoid metrics log noise when idle

commit 1a789ae76695bca801b55c9d4531e5438a3378c4
Author: Russell Bryant <rbryant@redhat.com>
Date:   Thu Sep 26 20:56:41 2024 +0000

    [Core] Avoid metrics log noise when idle
    
    Prior to this change, there was always a log message every 10 seconds,
    even when the system is idle. The noise when the metrics are all zeros
    doesn't seem very interesting, so this change makes it so the log
    message is only emitted while active. It will log a message whenever
    metrics are non-zero, or when returning to idle from non-idle.
    Repeated messages during idle now go to DEBUG instead of INFO.
    
    Signed-off-by: Russell Bryant <rbryant@redhat.com>


## Comments

### github-actions @ 2024-09-26T20:58:32Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### vrdn-23 @ 2024-10-24T02:53:29Z

@russellb I think this is a great small QoL improvement. Is this still something get worked on?
---
### russellb @ 2024-10-24T14:39:33Z

> @russellb I think this is a great small QoL improvement. Is this still something get worked on?

It's ready for review when someone has time.
---
### vrdn-23 @ 2024-10-24T16:57:44Z

@simon-mo @mgoin Is this something we can merge before the next release?
---
### vrdn-23 @ 2024-10-24T17:34:53Z

> I personally like keeping the log as a sort of "heartbeat" to let me know the server is functioning fine even if there is nothing running

I think the functionality of `/health` endpoint should be what indicates this rather than a log message showing idle numbers.
---
### russellb @ 2024-10-24T19:41:41Z

> > I personally like keeping the log as a sort of "heartbeat" to let me know the server is functioning fine even if there is nothing running
> 
> I think the functionality of `/health` endpoint should be what indicates this rather than a log message showing idle numbers.

A compromise could be to emit them as DEBUG instead of INFO once the system goes idle.
---
### vrdn-23 @ 2024-10-25T00:43:46Z

> A compromise could be to emit them as DEBUG instead of INFO once the system goes idle.

That sounds like a valid solution to me. Should also enable users who would prefer to have the logs enable it. Thoughts @mgoin ?
---
### russellb @ 2024-10-25T19:13:51Z

> > A compromise could be to emit them as DEBUG instead of INFO once the system goes idle.
> 
> That sounds like a valid solution to me. Should also enable users who would prefer to have the logs enable it. Thoughts @mgoin ?

I updated the PR to do this.
---
### vrdn-23 @ 2024-11-15T20:23:23Z

@mgoin can we go ahead with this approach?
---
### comaniac @ 2024-11-15T23:36:05Z

Just to check, do we have other DEBUG logging? Like if I enabled DEBUG would I see lots of other logs and even slow the performance?
---
### russellb @ 2024-11-19T16:31:36Z

> Just to check, do we have other DEBUG logging? Like if I enabled DEBUG would I see lots of other logs and even slow the performance?

It looks like `logger.debug()` is called in 120 places right now. 

```
$ git grep 'logger.debug(' | wc -l
120
```

On an idle system, I get all of this every 10 seconds (`vllm serve <model> --tensor-parallel-size 4`):

```
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:44 client.py:186] Waiting for output from MQLLMEngine.
DEBUG 11-19 16:29:51 client.py:165] Heartbeat successful.
INFO 11-19 16:29:51 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
DEBUG 11-19 16:29:51 client.py:165] Heartbeat successful.
DEBUG 11-19 16:29:51 engine.py:190] Waiting for new requests in engine loop.
```

When I hit the openai API server with a simple generate request, I got zero additional debug messages (just a few INFO messages).
---
### comaniac @ 2024-11-19T17:13:57Z

Thanks for the confirmation. From the log it seems not necessary to keep the message even in DEBUG level, because this level already has heartbeat logs. But I'll just leave this comment for reference and won't block the PR for merging.
---
### russellb @ 2024-11-19T17:19:58Z

> Thanks for the confirmation. From the log it seems not necessary to keep the message even in DEBUG level, because this level already has heartbeat logs. But I'll just leave this comment for reference and won't block the PR for merging.

Yeah, that's true. I'm happy with whatever. My main thing was that I think the default production config should have a quiet log when idle.
---
### stmcginnis @ 2024-11-19T17:40:55Z

> My main thing was that I think the default production config should have a quiet log when idle.

I would like to second this. I have several instances running with log shipping to a central place, and it is causing a lot of useless log messages to be collected. If we can reduce at least some of the noise, that would be ideal.
---
### vrdn-23 @ 2025-03-24T04:18:41Z

@russellb I realize this might not matter in the grand scheme of things going on, but is this something we can also implement in the V1 engine logs?
---
### russellb @ 2025-03-24T15:26:24Z

> @russellb I realize this might not matter in the grand scheme of things going on, but is this something we can also implement in the V1 engine logs?

Yeah, I'd personally like it in V1, as well. I just haven't prioritized it, yet. Feel free to port it over if you're interested!
---
### russellb @ 2025-04-16T17:04:42Z

a port of these changes to V1: https://github.com/vllm-project/vllm/pull/16735
---