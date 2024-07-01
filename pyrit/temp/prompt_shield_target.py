import requests
import os
import logging
import uuid
import duckdb

from typing import Any, Coroutine, Literal, Union, List
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface, DuckDBMemory

from pyrit.common import default_values
from pyrit.common import net_utility as nu

from pyrit.models import (
    data_serializer_factory, 
    construct_response_from_request,
    PromptRequestResponse, 
    PromptRequestPiece,
    Score
)

from pyrit.score import (
    Scorer
)

logger = logging.getLogger(__name__)

PromptShieldEntryKind = Literal["userPrompt", "document"]




###### README ######
'''
NOTE about Prompt Shield's HTTP body formatting:
Prompt Shield expects the following for its body (as JSON):
{
    {'userPrompt'}: str,
    {'documents'}: List[str]
}

And it returns the following in its response:
{
    {'userPromptAnalysis'}: {'attackDetected': bool},
    {'documentsAnalysis'}: List[{'attackDetected': bool}]
}

To make sure that scoring and prompt entries are consistent in the database,
the current implementation assumes one PromptRequestPiece per piece of data
sent to PromptShield. That is,
1 PromptRequestPiece == 1 Prompt Entry == 1 document XOR 1 user prompt == 1 Scoring Entry

This may not be the desired behavior long-term. For example, someone could want to send:
1 PromptRequestPiece == 1 HTTP request == * entries (e.g. 1 userPrompt AND 4 documents)
since that would be a more common type of request made to Prompt Shield. For now, because
the document and user classifiers operate independently and are different models, the
implementation is that every request to Prompt Shield contains exactly one document or
one user prompt.
'''




##### TO DO LIST ######
'''
TODO List MVP V1
-> Handle internal conversion in Scorer, Target for .original_value and .converted_value:
for target, extract the entire HTTP body
for score, extract only the attack detection (true/false)
-> Fix scoring issues (foreign key violated)
-> Make and run some unit tests
-> Allow for multiple PRpieces in one PRrequest
-> Incorporate the data_type serializer (just text-to-text, but so it plays nicely with
other infra)

TODO List MVP V2
-> DuckDB to Pandas helper methods (there is a duckdb library that already implements
db to pandas. Make a singleton class with visualization methods to make this easier to use.
maybe PandasInterface(metaclass='singleton')?)

TODO Bonus Round
-> Azure Blob storage PoC
'''




###### HELPER FUNCTIONS ######
def convert_entry_to_PRPiece(entry: str, kind: PromptShieldEntryKind) -> PromptRequestPiece:
    if kind not in ["userPrompt", "document"]:
        raise ValueError(f"Kind {kind} is not valid for Prompt Shield. Remember to pass each document in separately.")
    
    return PromptRequestPiece(
        role='user',
        original_value=entry,
        prompt_metadata=kind
    )

###### CLASSES ######
class PromptShieldTarget(PromptTarget):
    '''
    Prompt Shield as a Target. This uses the same logic as the Prompt Shield Scorer.
    TODO: Error catching. This includes checking for the most up-to-date API version for Prompt Shield,
    which is passed as parameters (params) in the HTTP request.
    TODO: Unit tests.
    TODO: The current implementation makes you specify which field you want to use when creating the
    target, since Prompt Shield has both a userPrompt and a document field.
    This doesn't account for cases in which you want to split the input across the two fields.
    I'm not sure how to implement this in orchestration, because every orchestrator has a slightly
    different flow that might affect the field(s) usage differently.
    '''

    ### ATTRIBUTES ###
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_ENDPOINT"
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_KEY"
    _endpoint: str
    _api_key: str
    _field: PromptShieldEntryKind

    ### METHODS ###
    def __init__(
            self,
            endpoint: str,
            api_key: str,
            field: PromptShieldEntryKind = 'document',
            memory: Union[MemoryInterface, None] = None
        ) -> None:

        super().__init__(memory=memory)

        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        self._field = field

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)

        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        headers = {
            'Ocp-Apim-Subscription-Key': self._api_key,
            'Content-Type': 'application/json',
        }

        params = {
            'api-version': '2024-02-15-preview',
        }

        # You need to send something for every field, even if said field is empty.
        userPromptValue: str = ""
        documentsValue: List[str] = [""]

        # match request.prompt_metadata:
        match self._field:
            case 'userPrompt':
                userPromptValue = request.converted_value
            case 'document':
                documentsValue = [request.converted_value]

        body = {
            'userPrompt': userPromptValue,
            'documents': documentsValue
        }

        response = await nu.make_request_and_raise_if_error_async(
            endpoint_uri=f'{self._endpoint}/contentsafety/text:shieldPrompt',
            method='POST',
            params=params,
            headers=headers,
            request_body=body
        )

        logger.info("Received a valid response from the prompt target")

        data = response.content

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(data)], response_type="text"
        )

        return response_entry

    
    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) == 0:
            raise ValueError("This target requires at least one prompt request piece.")
        if len(prompt_request.request_pieces) > 1:
            raise ValueError(
                    "Sorry, but requests with multiple entries are not supported yet. " \
                    "Please wrap each PromptRequestPiece in a PromptRequestResponse." \
                )
        
        # for prompt_request_piece in prompt_request.request_pieces:
        #     print("DEBUG:")
        #     print(prompt_request_piece)
        #     print(prompt_request_piece.converted_value)
        #     print(prompt_request_piece.conversation_id)
        #     match prompt_request_piece.prompt_metadata:
        #         case 'userPrompt':
        #             pass
        #         case 'document':
        #             pass
        #         case _:
        #             raise ValueError(
        #                 "The metadata for a PromptRequestPiece to Prompt Shield must indicate " \
        #                 "the actual role for the endpoint. It can be either 'document' or 'userPrompt'. " \
        #                 f"Got : {prompt_request_piece.prompt_metadata}."
        #             )

class PromptShieldScorer(Scorer):
    '''
    TODO list
    TODO: Implementation (Foreign Key Errors)
    TODO: Local testing
    TODO: Unit testing

    (Thanks to Richard Lundeen for this idea!)
    A scorer which returns a boolean value for detection by Prompt Shield.
    Since there's one scorer entry per prompt entry:
    (1 Scorer Entry == 1 PromptPiece == 1 HTTP Request to Prompt Shield)
    The score applies to whichever field (userPrompt or document/s) was sent in the
    PromptRequestPiece.

    NOTE: The HTTP body as a JSON returns a boolean value, and the intent is for it
    to be stored as a boolean value in the scoring table. If it can't, it will be
    converted to a string literal.
    '''

    ### ATTRIBUTES ###
    scorer_type: str
    _conversation_id: str
    _memory: Union[MemoryInterface, None]
    _target: PromptShieldTarget

    ### METHODS ###
    def __init__(
            self,
            target: PromptShieldTarget,
            memory: Union[MemoryInterface, None] = None
        ) -> None:
        '''
        TODO: description
        '''
        
        self.scorer_type = "true_false"
        self._conversation_id = str(uuid.uuid4())
        self._memory = memory if memory else DuckDBMemory()
        self._target: PromptShieldTarget = target

    async def score_async(self, request_response: PromptRequestPiece) -> List[Score]:
        '''
        NOTE: Use this as a debugging entry point
        TODO: description
        '''
        self.validate(request_response=request_response)

        # TODO: Fix this. The converted value should be the boolean for attack detection,
        # while the original would be the JSON response from the HTTP body.

        # body = request_response.converted_value
        body = request_response.original_value

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=body,
                    prompt_metadata=request_response.prompt_metadata,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._target.get_identifier()
                )
            ]
        )

        response = await self._target.send_prompt_async(prompt_request=request)

        # TODO: Fix this. The converted value should be the boolean for attack detection,
        # while the original would be the JSON response from the HTTP body.

        # result = response.request_pieces[0].converted_value

        # NOTE: This is a hacky workaround for the MVP. This needs to be fixed
        # in the PromptShieldTarget.

        result = str(response.request_pieces[0].original_value).split('"attackDetected":')[-1].split("}]}")[0]
        
        score = Score(
            score_type='true_false',
            score_value=result,
            score_value_description=None,
            score_category='attack_detection',
            score_metadata=None,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id # TODO: this is causing errors
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]
    
    def validate(
            self, 
            request_response: PromptRequestPiece
        ) -> None:
        '''
        NOTE: Use this as a debugging entry point
        TODO: implementation
        TODO: description
        '''
        pass



# class DuckDBDataAnalysisHelpers(metaclass='Singleton'):
#     dataset: DuckDB
#     def __init__() -> None:
#         ...
#     ...