from copy import deepcopy
import pytest
from test_vision_jobs import api
from test_vision_recovery import recovery
from test_vision_control import controlled
from laguna_gateway.thinking_selector import top_level_thinking
from laguna_gateway.engine import GatewayError


@pytest.mark.parametrize('value,want',[(True,True),(False,False),({'type':'enabled'},True),({'type':'disabled'},False)])
def test_public_and_native_selector_translation(api,value,want):
    _,_,app=api
    body={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':'hello'}], 'thinking':value}
    before=deepcopy(body)
    normalized=app.state.engine.normalise_request(body)[0]
    assert normalized['reasoning_effort']==('max' if want else 'none')
    assert normalized['chat_template_kwargs']['thinking'] is want
    assert 'thinking' not in normalized
    assert body==before


@pytest.mark.parametrize('kwargs',[{'thinking':True},{'enable_thinking':True},{'thinking_mode':'enabled'}])
def test_established_kwargs_win(api,kwargs):
    _,_,app=api
    body={'model':'deepseek-v4.1-flash','messages':[{'role':'user','content':'hello'}],
          'thinking':{'type':'disabled'},'chat_template_kwargs':kwargs}
    assert app.state.engine.normalise_request(body)[0]['reasoning_effort']=='max'


@pytest.mark.parametrize('value',[None,'off',{}, {'type':'unknown'},1])
def test_invalid_selector_has_explicit_error(api,value):
    _,_,app=api
    body={'messages':[{'role':'user','content':'hello'}],'thinking':value}
    with pytest.raises(GatewayError) as caught:app.state.engine.normalise_request(body)
    assert caught.value.code=='invalid_thinking_selector'


def test_absence_is_inert():
    assert top_level_thinking({}, {}) is None


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('switch',[True,False,{'type':'enabled'},{'type':'disabled'}])
def test_inline_image_preserves_mandatory_max_vision_profile(recovery,stream,switch):
    from test_vision_agent import agent
    client,_,_,native=recovery
    response=client.post('/v1/chat/completions',json=agent(thinking=switch,stream=stream))
    assert response.status_code==200,response.text
    sent=native.posts[0]
    assert (sent['reasoning_effort'],sent['thinking_budget'],sent['max_tokens'])==('max',8192,40960)
